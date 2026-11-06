//! Safe wrapper over libllama for the pieces a pipeline node needs.
//!
//! The vendored llama.cpp carries a small "layer range" patch: a model can be
//! loaded with only layers `[layer_begin, layer_end)` resident. Such a partial
//! model behaves like this:
//!
//! * `layer_begin == 0`: the piece owns the token embedding, so it accepts tokens.
//!   Otherwise it accepts hidden states (`decode_embd`).
//! * `layer_end == n_layer`: the piece owns the final norm and LM head, so it
//!   produces logits. Otherwise it produces the raw residual-stream hidden state
//!   at the boundary (`embeddings()`), one `n_embd` row per output token.
//!
//! A full model is simply the range `[0, n_layer)`.
//!
//! Thread safety: a [`Model`] is immutable after load and may be shared across
//! threads. A [`Context`] is `Send` but not `Sync`; wrap it in a mutex.

use dfin_llama_sys as sys;
use std::ffi::{c_char, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Once};

pub type Token = sys::llama_token;
pub type Pos = sys::llama_pos;
pub type SeqId = sys::llama_seq_id;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load model {0}")]
    ModelLoad(String),
    #[error("failed to create llama context")]
    ContextInit,
    #[error("llama_decode failed with status {0}")]
    Decode(i32),
    #[error("llama_decode: no free KV slot (context full)")]
    ContextFull,
    #[error("llama_decode aborted")]
    Aborted,
    #[error("tokenization failed")]
    Tokenize,
    #[error("chat template application failed (template unsupported?)")]
    ChatTemplate,
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("the vendored libllama lacks the layer-range patch; cannot load a partial model")]
    NoLayerRange,
}

pub type Result<T> = std::result::Result<T, Error>;

static INIT: Once = Once::new();

/// Initialise ggml/llama backends and route llama.cpp logging into `tracing`.
/// Safe to call repeatedly.
pub fn init() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log_callback), ptr::null_mut());
        sys::llama_backend_init();
    });
}

unsafe extern "C" fn log_callback(level: sys::ggml_log_level, text: *const c_char, _user: *mut c_void) {
    if text.is_null() {
        return;
    }
    let msg = CStr::from_ptr(text).to_string_lossy();
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    // Head and stage contexts deliberately request pooling NONE to read per-token hidden
    // states; llama.cpp warns about overriding the (unspecified) model default every time.
    if msg.contains("pooling_type is [-1], but [0] was specified") {
        tracing::trace!(target: "llama", "{msg}");
        return;
    }
    match level {
        sys::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama", "{msg}"),
        sys::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama", "{msg}"),
        sys::GGML_LOG_LEVEL_INFO => tracing::debug!(target: "llama", "{msg}"),
        _ => tracing::trace!(target: "llama", "{msg}"),
    }
}

/// Whether this build can load partial (layer-range) models.
pub fn has_layer_range() -> bool {
    sys::HAS_LAYER_RANGE
}

pub fn system_info() -> String {
    init();
    unsafe { CStr::from_ptr(sys::llama_print_system_info()).to_string_lossy().into_owned() }
}

pub fn supports_gpu_offload() -> bool {
    init();
    unsafe { sys::llama_supports_gpu_offload() }
}

/// A compute device ggml knows about (CPU, GPUs, accelerators).
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub is_gpu: bool,
    pub memory_free: usize,
    pub memory_total: usize,
}

pub fn devices() -> Vec<DeviceInfo> {
    init();
    let mut out = Vec::new();
    unsafe {
        let n = sys::ggml_backend_dev_count();
        for i in 0..n {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let (mut free, mut total) = (0usize, 0usize);
            sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
            let ty = sys::ggml_backend_dev_type(dev);
            out.push(DeviceInfo {
                name: cstr_to_string(sys::ggml_backend_dev_name(dev)),
                description: cstr_to_string(sys::ggml_backend_dev_description(dev)),
                is_gpu: ty == sys::GGML_BACKEND_DEVICE_TYPE_GPU || ty == sys::GGML_BACKEND_DEVICE_TYPE_IGPU,
                memory_free: free,
                memory_total: total,
            });
        }
    }
    out
}

unsafe fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

// ----------------------------------------------------------------------------------------------
// Model

/// How to load a model (or a piece of one).
#[derive(Clone, Debug)]
pub struct ModelParams {
    /// Layers to offload to the GPU; `-1` means all of this piece's layers.
    pub n_gpu_layers: i32,
    /// First layer (inclusive) owned by this piece. `0` means it also owns the embedding.
    pub layer_begin: i32,
    /// One past the last layer. `-1` means `n_layer`, i.e. this piece owns the LM head.
    pub layer_end: i32,
    /// Memory-map the file instead of reading it.
    pub use_mmap: bool,
    /// Lock the weights in RAM at load (with `use_mmap`, forces every page resident so the
    /// first request does not pay a page-fault storm). Needs enough free memory.
    pub use_mlock: bool,
    /// Regex patterns of tensor names to keep in CPU memory even when offloading,
    /// e.g. `"exps"` keeps routed experts in RAM (the `-ot exps=CPU` pattern).
    pub cpu_tensor_patterns: Vec<String>,
    /// Load only the vocabulary; no weights. For tokenizer-only uses.
    pub vocab_only: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        ModelParams {
            n_gpu_layers: -1,
            layer_begin: 0,
            layer_end: -1,
            use_mmap: true,
            use_mlock: false,
            cpu_tensor_patterns: Vec::new(),
            vocab_only: false,
        }
    }
}

impl ModelParams {
    pub fn full() -> Self {
        Self::default()
    }

    pub fn range(layer_begin: i32, layer_end: i32) -> Self {
        ModelParams {
            layer_begin,
            layer_end,
            ..Self::default()
        }
    }

    pub fn is_partial_request(&self) -> bool {
        self.layer_begin != 0 || self.layer_end != -1
    }
}

/// A loaded model or model piece. Immutable after load; share with `Arc`.
pub struct Model {
    ptr: *mut sys::llama_model,
    path: PathBuf,
    layer_begin: i32,
    layer_end: i32,
    n_layer: i32,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::llama_model_free(self.ptr) }
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model({}, layers {}..{} of {})",
            self.path.display(),
            self.layer_begin,
            self.layer_end,
            self.n_layer
        )
    }
}

impl Model {
    pub fn load(path: &Path, params: &ModelParams) -> Result<Model> {
        init();
        if params.is_partial_request() && !sys::HAS_LAYER_RANGE {
            return Err(Error::NoLayerRange);
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::Invalid("model path contains NUL".into()))?;

        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = params.n_gpu_layers;
        mp.vocab_only = params.vocab_only;
        mp.load_mode = match (params.use_mmap, params.use_mlock) {
            (true, true) => sys::LLAMA_LOAD_MODE_MMAP_MLOCK,
            (true, false) => sys::LLAMA_LOAD_MODE_MMAP,
            (false, true) => sys::LLAMA_LOAD_MODE_MLOCK,
            (false, false) => sys::LLAMA_LOAD_MODE_NONE,
        };
        sys::set_layer_range(&mut mp, params.layer_begin, params.layer_end);

        // Tensor placement overrides: keep matching tensors on the CPU buffer type.
        let patterns: Vec<CString> = params
            .cpu_tensor_patterns
            .iter()
            .map(|p| CString::new(p.as_bytes()).map_err(|_| Error::Invalid("pattern contains NUL".into())))
            .collect::<Result<_>>()?;
        let mut overrides: Vec<sys::llama_model_tensor_buft_override> = Vec::new();
        if !patterns.is_empty() {
            let cpu_buft = unsafe { sys::ggml_backend_cpu_buffer_type() };
            for p in &patterns {
                overrides.push(sys::llama_model_tensor_buft_override {
                    pattern: p.as_ptr(),
                    buft: cpu_buft,
                });
            }
            overrides.push(sys::llama_model_tensor_buft_override {
                pattern: ptr::null(),
                buft: ptr::null_mut(),
            });
            mp.tensor_buft_overrides = overrides.as_ptr();
        }

        let ptr = unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), mp) };
        drop(overrides);
        drop(patterns);
        if ptr.is_null() {
            return Err(Error::ModelLoad(path.display().to_string()));
        }
        let n_layer = unsafe { sys::llama_model_n_layer(ptr) };
        let (layer_begin, layer_end) = unsafe { sys::model_layer_range(ptr) }.unwrap_or((0, n_layer));
        Ok(Model {
            ptr,
            path: path.to_path_buf(),
            layer_begin,
            layer_end,
            n_layer,
        })
    }

    pub fn as_ptr(&self) -> *mut sys::llama_model {
        self.ptr
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn n_embd(&self) -> i32 {
        unsafe { sys::llama_model_n_embd(self.ptr) }
    }

    /// Width of one input row: `n_embd` for a whole model or a head; a multiple of
    /// `n_embd` for a stage of an architecture whose state between layers is wider
    /// (hyper-connection streams, residual banks).
    pub fn n_embd_inp(&self) -> i32 {
        unsafe { sys::llama_model_n_embd_inp(self.ptr) }
    }

    /// Width of one output row of a stage without the LM head (see [`Model::n_embd_inp`]).
    pub fn n_embd_out(&self) -> i32 {
        unsafe { sys::llama_model_n_embd_out(self.ptr) }
    }

    pub fn n_layer(&self) -> i32 {
        self.n_layer
    }

    pub fn n_head(&self) -> i32 {
        unsafe { sys::llama_model_n_head(self.ptr) }
    }

    pub fn n_ctx_train(&self) -> i32 {
        unsafe { sys::llama_model_n_ctx_train(self.ptr) }
    }

    /// Bytes of weights actually resident for this piece.
    pub fn size_bytes(&self) -> u64 {
        unsafe { sys::llama_model_size(self.ptr) }
    }

    /// Resolved `[begin, end)` layer range of this piece.
    pub fn layer_range(&self) -> (i32, i32) {
        (self.layer_begin, self.layer_end)
    }

    pub fn is_partial(&self) -> bool {
        self.layer_begin != 0 || self.layer_end != self.n_layer
    }

    /// Owns the token embedding: accepts tokens as input.
    pub fn has_head(&self) -> bool {
        self.layer_begin == 0
    }

    /// Owns the final norm and LM head: produces logits.
    pub fn has_tail(&self) -> bool {
        self.layer_end == self.n_layer
    }

    pub fn desc(&self) -> String {
        let mut buf = vec![0u8; 256];
        let n = unsafe { sys::llama_model_desc(self.ptr, buf.as_mut_ptr() as *mut c_char, buf.len()) };
        if n < 0 {
            return String::new();
        }
        buf.truncate((n as usize).min(buf.len() - 1));
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// A GGUF metadata value, e.g. `general.name` or `general.architecture`.
    pub fn meta_str(&self, key: &str) -> Option<String> {
        let c_key = CString::new(key).ok()?;
        let mut buf = vec![0u8; 4096];
        let n = unsafe {
            sys::llama_model_meta_val_str(self.ptr, c_key.as_ptr(), buf.as_mut_ptr() as *mut c_char, buf.len())
        };
        if n < 0 {
            return None;
        }
        buf.truncate((n as usize).min(buf.len() - 1));
        Some(String::from_utf8_lossy(&buf).into_owned())
    }

    /// The model's built-in chat template (Jinja source), if the GGUF has one.
    pub fn chat_template(&self) -> Option<String> {
        unsafe {
            let p = sys::llama_model_chat_template(self.ptr, ptr::null());
            if p.is_null() {
                None
            } else {
                Some(CStr::from_ptr(p).to_string_lossy().into_owned())
            }
        }
    }

    pub fn vocab(&self) -> Vocab<'_> {
        Vocab {
            ptr: unsafe { sys::llama_model_get_vocab(self.ptr) },
            _model: PhantomData,
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Vocab

/// Tokenizer and special-token queries. Borrowed from a [`Model`].
#[derive(Clone, Copy)]
pub struct Vocab<'a> {
    ptr: *const sys::llama_vocab,
    _model: PhantomData<&'a Model>,
}

impl<'a> Vocab<'a> {
    pub fn n_tokens(&self) -> i32 {
        unsafe { sys::llama_vocab_n_tokens(self.ptr) }
    }

    pub fn bos(&self) -> Token {
        unsafe { sys::llama_vocab_bos(self.ptr) }
    }

    pub fn eos(&self) -> Token {
        unsafe { sys::llama_vocab_eos(self.ptr) }
    }

    /// End-of-generation: EOS, EOT or any other turn-ending control token.
    pub fn is_eog(&self, token: Token) -> bool {
        unsafe { sys::llama_vocab_is_eog(self.ptr, token) }
    }

    pub fn add_bos(&self) -> bool {
        unsafe { sys::llama_vocab_get_add_bos(self.ptr) }
    }

    /// Tokenize UTF-8 text. `add_special` prepends BOS when the model wants one;
    /// `parse_special` lets control tokens in the text (e.g. `<|im_start|>`) be
    /// recognised instead of being split into bytes.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<Token>> {
        let mut cap = text.len() + 8;
        loop {
            let mut out = vec![0 as Token; cap];
            let n = unsafe {
                sys::llama_tokenize(
                    self.ptr,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    out.as_mut_ptr(),
                    cap as i32,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                out.truncate(n as usize);
                return Ok(out);
            }
            let needed = (-n) as usize;
            if needed <= cap {
                return Err(Error::Tokenize);
            }
            cap = needed;
        }
    }

    /// Raw bytes of one token's text. May be a partial UTF-8 sequence for byte tokens.
    pub fn token_to_piece(&self, token: Token, special: bool) -> Vec<u8> {
        let mut cap = 64usize;
        loop {
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                sys::llama_token_to_piece(self.ptr, token, buf.as_mut_ptr() as *mut c_char, cap as i32, 0, special)
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return buf;
            }
            cap = (-n) as usize;
        }
    }

    pub fn detokenize(&self, tokens: &[Token], remove_special: bool, unparse_special: bool) -> String {
        let mut cap = tokens.len() * 8 + 16;
        loop {
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                sys::llama_detokenize(
                    self.ptr,
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    buf.as_mut_ptr() as *mut c_char,
                    cap as i32,
                    remove_special,
                    unparse_special,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return String::from_utf8_lossy(&buf).into_owned();
            }
            cap = (-n) as usize;
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Chat templates

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render a conversation with llama.cpp's built-in template engine. `template` is
/// either a template name (`"chatml"`, `"llama3"`, ...) or Jinja source from
/// [`Model::chat_template`]; llama.cpp recognises common templates by content.
pub fn chat_apply_template(template: &str, messages: &[ChatMessage], add_assistant: bool) -> Result<String> {
    let c_tmpl = CString::new(template).map_err(|_| Error::Invalid("template contains NUL".into()))?;
    let c_msgs: Vec<(CString, CString)> = messages
        .iter()
        .map(|m| {
            Ok((
                CString::new(m.role.as_bytes()).map_err(|_| Error::Invalid("role contains NUL".into()))?,
                CString::new(m.content.as_bytes()).map_err(|_| Error::Invalid("content contains NUL".into()))?,
            ))
        })
        .collect::<Result<_>>()?;
    let raw: Vec<sys::llama_chat_message> = c_msgs
        .iter()
        .map(|(r, c)| sys::llama_chat_message {
            role: r.as_ptr(),
            content: c.as_ptr(),
        })
        .collect();
    let mut cap: usize = messages.iter().map(|m| m.content.len() + m.role.len() + 32).sum::<usize>() + 64;
    loop {
        let mut buf = vec![0u8; cap];
        let n = unsafe {
            sys::llama_chat_apply_template(
                c_tmpl.as_ptr(),
                raw.as_ptr(),
                raw.len(),
                add_assistant,
                buf.as_mut_ptr() as *mut c_char,
                cap as i32,
            )
        };
        if n < 0 {
            return Err(Error::ChatTemplate);
        }
        if (n as usize) <= cap {
            buf.truncate(n as usize);
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }
        cap = n as usize + 1;
    }
}

// ----------------------------------------------------------------------------------------------
// Context

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashAttn {
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
}

impl KvType {
    fn to_ggml(self) -> sys::ggml_type {
        match self {
            KvType::F32 => sys::GGML_TYPE_F32,
            KvType::F16 => sys::GGML_TYPE_F16,
            KvType::BF16 => sys::GGML_TYPE_BF16,
            KvType::Q8_0 => sys::GGML_TYPE_Q8_0,
            KvType::Q4_0 => sys::GGML_TYPE_Q4_0,
        }
    }
}

impl std::str::FromStr for KvType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Ok(KvType::F32),
            "f16" => Ok(KvType::F16),
            "bf16" => Ok(KvType::BF16),
            "q8_0" => Ok(KvType::Q8_0),
            "q4_0" => Ok(KvType::Q4_0),
            other => Err(format!("unknown KV cache type {other:?}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextParams {
    /// Total KV capacity in tokens, shared by all sequences.
    pub n_ctx: u32,
    /// Max tokens per `decode` call.
    pub n_batch: u32,
    /// Physical micro-batch size.
    pub n_ubatch: u32,
    /// Distinct sequences (sessions) the context can hold at once.
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    /// Output hidden states (`embeddings()`) rather than only logits. Required for
    /// head and middle pieces; a tail piece or a full model leaves this `false`.
    pub embeddings: bool,
    pub flash_attn: FlashAttn,
    pub type_k: KvType,
    pub type_v: KvType,
    /// One shared KV buffer across sequences. Leave `false` with `n_seq_max > 1`.
    pub kv_unified: bool,
    pub offload_kqv: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let threads = (cores / 2).max(1) as i32;
        ContextParams {
            n_ctx: 4096,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: threads,
            n_threads_batch: threads,
            embeddings: false,
            flash_attn: FlashAttn::Auto,
            type_k: KvType::F16,
            type_v: KvType::F16,
            kv_unified: false,
            offload_kqv: true,
        }
    }
}

/// Which batch rows to produce outputs (hidden states or logits) for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// Every token. Required for head and middle pieces during prefill, since the
    /// next piece needs all of them.
    All,
    /// Only the last token (normal LM decode).
    Last,
    /// None: only extend the KV cache.
    None,
}

/// An inference context (KV cache plus compute graph) over a [`Model`].
pub struct Context {
    ptr: *mut sys::llama_context,
    model: Arc<Model>,
    n_embd: usize,
    n_embd_inp: usize,
    n_embd_out: usize,
    n_vocab: usize,
    last_n_outputs: usize,
}

unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::llama_free(self.ptr) }
    }
}

impl Context {
    pub fn new(model: Arc<Model>, params: &ContextParams) -> Result<Context> {
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = params.n_ctx;
        cp.n_batch = params.n_batch;
        cp.n_ubatch = params.n_ubatch.min(params.n_batch);
        cp.n_seq_max = params.n_seq_max.max(1);
        cp.n_threads = params.n_threads;
        cp.n_threads_batch = params.n_threads_batch;
        cp.embeddings = params.embeddings;
        if params.embeddings {
            // Per-token hidden states, no pooling. Left unspecified otherwise so
            // llama.cpp does not warn about overriding the model's default.
            cp.pooling_type = sys::LLAMA_POOLING_TYPE_NONE;
        }
        cp.flash_attn_type = match params.flash_attn {
            FlashAttn::Auto => sys::LLAMA_FLASH_ATTN_TYPE_AUTO,
            FlashAttn::On => sys::LLAMA_FLASH_ATTN_TYPE_ENABLED,
            FlashAttn::Off => sys::LLAMA_FLASH_ATTN_TYPE_DISABLED,
        };
        cp.type_k = params.type_k.to_ggml();
        cp.type_v = params.type_v.to_ggml();
        cp.kv_unified = params.kv_unified;
        cp.offload_kqv = params.offload_kqv;
        cp.no_perf = true;

        let ptr = unsafe { sys::llama_init_from_model(model.ptr, cp) };
        if ptr.is_null() {
            return Err(Error::ContextInit);
        }
        let n_embd = model.n_embd() as usize;
        let n_embd_inp = model.n_embd_inp().max(1) as usize;
        let n_embd_out = model.n_embd_out().max(1) as usize;
        let n_vocab = model.vocab().n_tokens() as usize;
        Ok(Context {
            ptr,
            model,
            n_embd,
            n_embd_inp,
            n_embd_out,
            n_vocab,
            last_n_outputs: 0,
        })
    }

    pub fn as_ptr(&self) -> *mut sys::llama_context {
        self.ptr
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Width of one hidden-state row this context takes with [`Context::decode_embd`].
    pub fn n_embd_inp(&self) -> usize {
        self.n_embd_inp
    }

    /// Width of one row of [`Context::embeddings`].
    pub fn n_embd_out(&self) -> usize {
        self.n_embd_out
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { sys::llama_n_ctx(self.ptr) }
    }

    pub fn n_seq_max(&self) -> u32 {
        unsafe { sys::llama_n_seq_max(self.ptr) }
    }

    pub fn set_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        unsafe { sys::llama_set_n_threads(self.ptr, n_threads, n_threads_batch) }
    }

    /// Feed tokens at positions `pos0 .. pos0 + tokens.len()` into sequence `seq`.
    /// Only valid on a piece that owns the embedding (`has_head()`).
    /// Returns the number of output rows produced.
    pub fn decode_tokens(&mut self, tokens: &[Token], pos0: Pos, seq: SeqId, output: OutputMode) -> Result<usize> {
        if !self.model.has_head() {
            return Err(Error::Invalid("this piece has no token embedding; use decode_embd".into()));
        }
        self.decode_inner(Some(tokens), None, tokens.len(), pos0, seq, output)
    }

    /// Feed hidden states (`n_tokens * n_embd` f32 values, token-major) at positions
    /// `pos0 ..` into sequence `seq`. Valid on any piece; on a head piece it bypasses
    /// the embedding table.
    pub fn decode_embd(&mut self, embd: &[f32], pos0: Pos, seq: SeqId, output: OutputMode) -> Result<usize> {
        if embd.len() % self.n_embd_inp != 0 {
            return Err(Error::Invalid(format!(
                "embedding buffer length {} is not a multiple of the input width {}",
                embd.len(),
                self.n_embd_inp
            )));
        }
        let n_tokens = embd.len() / self.n_embd_inp;
        self.decode_inner(None, Some(embd), n_tokens, pos0, seq, output)
    }

    fn decode_inner(
        &mut self,
        tokens: Option<&[Token]>,
        embd: Option<&[f32]>,
        n_tokens: usize,
        pos0: Pos,
        seq: SeqId,
        output: OutputMode,
    ) -> Result<usize> {
        if n_tokens == 0 {
            return Err(Error::Invalid("empty batch".into()));
        }
        let embd_dim = if embd.is_some() { self.n_embd_inp as i32 } else { 0 };
        let mut batch = unsafe { sys::llama_batch_init(n_tokens as i32, embd_dim, 1) };
        let mut n_outputs = 0usize;
        unsafe {
            batch.n_tokens = n_tokens as i32;
            for i in 0..n_tokens {
                if let Some(t) = tokens {
                    *batch.token.add(i) = t[i];
                }
                *batch.pos.add(i) = pos0 + i as Pos;
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)).add(0) = seq;
                let out = match output {
                    OutputMode::All => true,
                    OutputMode::Last => i + 1 == n_tokens,
                    OutputMode::None => false,
                };
                *batch.logits.add(i) = out as i8;
                n_outputs += out as usize;
            }
            if let Some(e) = embd {
                ptr::copy_nonoverlapping(e.as_ptr(), batch.embd, n_tokens * self.n_embd_inp);
            }
        }
        let rc = unsafe { sys::llama_decode(self.ptr, batch) };
        unsafe { sys::llama_batch_free(batch) };
        match rc {
            0 => {}
            1 => return Err(Error::ContextFull),
            2 => return Err(Error::Aborted),
            n => return Err(Error::Decode(n)),
        }
        self.last_n_outputs = n_outputs;
        Ok(n_outputs)
    }

    /// Hidden states for the output rows of the last decode: `n_outputs * n_embd`
    /// f32 values, token-major. Requires `ContextParams::embeddings`. On a partial
    /// piece these are the raw residual-stream values at the stage boundary.
    pub fn embeddings(&self) -> &[f32] {
        unsafe {
            let p = sys::llama_get_embeddings(self.ptr);
            if p.is_null() || self.last_n_outputs == 0 {
                return &[];
            }
            std::slice::from_raw_parts(p, self.last_n_outputs * self.n_embd_out)
        }
    }

    /// Logits for the output rows of the last decode: `n_outputs * n_vocab` values.
    /// Only a piece with the LM head (`has_tail()`) produces logits.
    pub fn logits(&self) -> &[f32] {
        unsafe {
            let p = sys::llama_get_logits(self.ptr);
            if p.is_null() || self.last_n_outputs == 0 {
                return &[];
            }
            std::slice::from_raw_parts(p, self.last_n_outputs * self.n_vocab)
        }
    }

    /// Logits of output row `i` (negative indexes from the end, `-1` = last).
    pub fn logits_ith(&self, i: i32) -> &[f32] {
        unsafe {
            let p = sys::llama_get_logits_ith(self.ptr, i);
            if p.is_null() {
                return &[];
            }
            std::slice::from_raw_parts(p, self.n_vocab)
        }
    }

    pub fn last_n_outputs(&self) -> usize {
        self.last_n_outputs
    }

    /// Remove KV entries of `seq` with `p0 <= pos < p1` (`p1 = -1`: to the end,
    /// `p0 = -1`: from the start). Returns `false` if the removal is not possible.
    pub fn seq_rm(&mut self, seq: SeqId, p0: Pos, p1: Pos) -> bool {
        unsafe { sys::llama_memory_seq_rm(sys::llama_get_memory(self.ptr), seq, p0, p1) }
    }

    /// Copy KV entries of `src` in `[p0, p1)` to `dst` (prefix sharing / forking).
    pub fn seq_cp(&mut self, src: SeqId, dst: SeqId, p0: Pos, p1: Pos) {
        unsafe { sys::llama_memory_seq_cp(sys::llama_get_memory(self.ptr), src, dst, p0, p1) }
    }

    /// Highest position stored for `seq`, or `-1` if empty.
    pub fn seq_pos_max(&self, seq: SeqId) -> Pos {
        unsafe { sys::llama_memory_seq_pos_max(sys::llama_get_memory(self.ptr), seq) }
    }

    /// Drop all KV entries.
    pub fn clear(&mut self) {
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ptr), true) }
    }

    /// Wait for any in-flight GPU work.
    pub fn synchronize(&mut self) {
        unsafe { sys::llama_synchronize(self.ptr) }
    }
}
