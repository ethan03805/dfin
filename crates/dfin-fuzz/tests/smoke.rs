//! The fuzz targets as ordinary tests, on every platform and toolchain: each target
//! over its seeds, every truncation of every seed, byte and bit flips of every seed,
//! random bytes, random bytes behind a valid prefix, and whatever a cargo-fuzz run left
//! in `fuzz/corpus` and `fuzz/artifacts`. A panic anywhere fails the test and names
//! the target and the input.
//!
//! `DFIN_FUZZ_ITERS` scales the random part (default 5000 per target; a debug build
//! spends most of its time in signature checks, so CI runs this in release).

use dfin_fuzz::{seeds, Target, TARGETS};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use std::path::{Path, PathBuf};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Run one input, naming it if the target panics.
fn run(name: &str, target: Target, what: &str, input: &[u8]) -> bool {
    let shown = if input.len() <= 96 {
        hex(input)
    } else {
        format!("{}... ({} bytes)", hex(&input[..96]), input.len())
    };
    let r = std::panic::catch_unwind(|| target(input));
    match r {
        Ok(decoded) => decoded,
        Err(_) => panic!("target {name} panicked on {what}: {shown}"),
    }
}

fn corpus_dirs(name: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz");
    vec![root.join("corpus").join(name), root.join("artifacts").join(name)]
}

#[test]
fn every_target_survives_seeds_mutations_and_noise() {
    let iters: usize = std::env::var("DFIN_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000);
    let mut rng = StdRng::seed_from_u64(0x5eed);
    for (name, target) in TARGETS {
        let started = std::time::Instant::now();
        let seeds = seeds::for_target(name);
        assert!(!seeds.is_empty(), "{name} has no seeds");
        let mut decoded = 0usize;
        let mut rejected = 0usize;
        let mut count = |ok: bool| if ok { decoded += 1 } else { rejected += 1 };

        for (i, seed) in seeds.iter().enumerate() {
            assert!(run(name, *target, &format!("seed {i}"), seed), "{name} seed {i} does not decode");
            // Every truncation.
            for cut in 0..seed.len() {
                count(run(name, *target, &format!("seed {i} cut at {cut}"), &seed[..cut]));
            }
            // Every byte replaced by a few values, and every bit flipped, for short seeds;
            // a sample of positions for long ones (the directory seeds verify signatures
            // on every variant).
            let positions: Vec<usize> = if seed.len() <= 192 {
                (0..seed.len()).collect()
            } else {
                (0..192).map(|_| rng.random_range(0..seed.len())).collect()
            };
            for &pos in &positions {
                for v in [0u8, 0xff, 0x80, 0x01, seed[pos].wrapping_add(1)] {
                    let mut m = seed.clone();
                    m[pos] = v;
                    count(run(name, *target, &format!("seed {i} byte {pos} = {v:#x}"), &m));
                }
                for bit in 0..8 {
                    let mut m = seed.clone();
                    m[pos] ^= 1 << bit;
                    count(run(name, *target, &format!("seed {i} bit {pos}.{bit}"), &m));
                }
            }
            // Spliced tail: the seed's first half followed by random bytes.
            for _ in 0..64 {
                let keep = rng.random_range(0..=seed.len());
                let mut m = seed[..keep].to_vec();
                let extra = rng.random_range(0..64);
                let mut tail = vec![0u8; extra];
                rng.fill_bytes(&mut tail);
                m.extend(tail);
                count(run(name, *target, &format!("seed {i} spliced at {keep}"), &m));
            }
        }
        // Pure noise, in lengths that stay cheap.
        for _ in 0..iters {
            let len = match rng.random_range(0..10) {
                0 => 0,
                1..=6 => rng.random_range(1..32),
                7..=8 => rng.random_range(32..512),
                _ => rng.random_range(512..4096),
            };
            let mut m = vec![0u8; len];
            rng.fill_bytes(&mut m);
            // Boundary bytes make long varints and extreme integers likelier; plain noise
            // almost never produces two 32-bit values near their limit at once.
            if rng.random_bool(0.5) {
                for b in m.iter_mut() {
                    if rng.random_bool(0.25) {
                        *b = [0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff][rng.random_range(0..6)];
                    }
                }
            }
            count(run(name, *target, "random bytes", &m));
        }
        // What libFuzzer left behind, if anything.
        for dir in corpus_dirs(name) {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                if let Ok(bytes) = std::fs::read(e.path()) {
                    count(run(name, *target, &e.path().display().to_string(), &bytes));
                }
            }
        }
        eprintln!(
            "{name}: {decoded} decoded, {rejected} rejected, {:.1} s",
            started.elapsed().as_secs_f32()
        );
        assert!(decoded > 0 && rejected > 0, "{name}: mutations should reach both outcomes");
    }
}

/// What libFuzzer found on its first run: the limit helper overflowed for widths a
/// hostile peer can advertise. It saturates at the frame cap now.
#[test]
fn frame_limit_saturates_at_the_frame_cap() {
    use dfin_proto::{forward_frame_limit, MAX_FRAME_BYTES};
    assert_eq!(forward_frame_limit(u32::MAX, u32::MAX), MAX_FRAME_BYTES);
    assert_eq!(forward_frame_limit(u32::MAX, 1), MAX_FRAME_BYTES);
    assert_eq!(forward_frame_limit(0, 0), 65536);
    assert_eq!(forward_frame_limit(2, 4), 32 + 65536);
}

/// The stage's cheap size check keys on the first body byte: no request but `Forward`
/// may start with the Forward tag, in any encoding the seeds produce.
#[test]
fn only_forward_carries_the_forward_tag() {
    for req in seeds::requests() {
        let bytes = postcard::to_stdvec(&req).unwrap();
        let is_forward = matches!(req, dfin_proto::Request::Forward { .. });
        assert_eq!(bytes[0] == dfin_proto::Request::FORWARD_TAG, is_forward, "{req:?}");
    }
}
