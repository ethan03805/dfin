//! Streaming text helpers: a UTF-8 accumulator for token pieces and a stop-string
//! matcher that holds back text which might be the start of a stop string.

/// Collects raw token bytes and releases only complete UTF-8 sequences.
#[derive(Default)]
pub struct Utf8Accumulator {
    buf: Vec<u8>,
}

impl Utf8Accumulator {
    /// Append bytes; return the longest decodable prefix. Invalid sequences are
    /// replaced with U+FFFD; an incomplete trailing sequence is carried over.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.buf[..valid]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            // Incomplete sequence at the end: keep it for next time.
                            self.buf.drain(..valid);
                            return out;
                        }
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.buf.drain(..valid + n);
                        }
                    }
                }
            }
        }
    }

    /// Release whatever is left, lossily.
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}

/// Result of feeding text into a [`StopMatcher`].
#[derive(Debug, PartialEq, Eq)]
pub struct StopPush {
    /// Text that is safe to emit now.
    pub emit: String,
    /// A stop string matched; generation should end and nothing after it is emitted.
    pub stopped: bool,
}

pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> StopPush {
        if self.stops.is_empty() {
            return StopPush {
                emit: text.to_string(),
                stopped: false,
            };
        }
        self.pending.push_str(text);
        // Earliest full match of any stop string.
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(i) = hit {
            let emit = self.pending[..i].to_string();
            self.pending.clear();
            return StopPush {
                emit,
                stopped: true,
            };
        }
        // Hold back the longest suffix that is a proper prefix of some stop string.
        let mut hold = 0usize;
        for s in &self.stops {
            let max = s.len().saturating_sub(1).min(self.pending.len());
            for len in (1..=max).rev() {
                let start = self.pending.len() - len;
                if !self.pending.is_char_boundary(start) {
                    continue;
                }
                if s.starts_with(&self.pending[start..]) {
                    hold = hold.max(len);
                    break;
                }
            }
        }
        let cut = self.pending.len() - hold;
        let emit = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        StopPush {
            emit,
            stopped: false,
        }
    }

    /// Release held-back text at the end of generation.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_split_sequence() {
        let mut a = Utf8Accumulator::default();
        let euro: &[u8] = &[0xE2, 0x82, 0xAC]; // U+20AC
        assert_eq!(a.push(b"a"), "a");
        assert_eq!(a.push(&euro[..1]), "");
        assert_eq!(a.push(&euro[1..2]), "");
        assert_eq!(a.push(&euro[2..]), "\u{20AC}");
        assert_eq!(a.push(&[0xE2]), "");
        assert_eq!(a.flush(), "\u{FFFD}");
    }

    #[test]
    fn utf8_invalid_byte() {
        let mut a = Utf8Accumulator::default();
        assert_eq!(a.push(&[b'x', 0xFF, b'y']), "x\u{FFFD}y");
    }

    #[test]
    fn stop_hold_back_and_match() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("hello E").emit, "hello ");
        let r = m.push("N");
        assert_eq!(
            r,
            StopPush {
                emit: String::new(),
                stopped: false
            }
        );
        let r = m.push("Dx");
        assert_eq!(
            r,
            StopPush {
                emit: String::new(),
                stopped: true
            }
        );
    }

    #[test]
    fn stop_false_alarm_released() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("E").emit, "");
        assert_eq!(m.push("x").emit, "Ex");
        assert_eq!(m.push("EN").emit, "");
        assert_eq!(m.flush(), "EN");
    }

    #[test]
    fn no_stops_passthrough() {
        let mut m = StopMatcher::new(vec![]);
        assert_eq!(m.push("abc").emit, "abc");
    }
}
