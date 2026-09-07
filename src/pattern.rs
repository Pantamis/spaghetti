//! Vanity prefix parsing, validation, compilation to x-coordinate bit masks,
//! and matching.
//!
//! Bit layout: the address payload is `ser_P(B_scan) ++ ser_P(B_m)` read
//! MSB-first in 5-bit groups after the version character `q`. Stream bits 0..6
//! are the constant `000000` and `1` of the SEC1 tag byte (`0x02`/`0x03`),
//! stream bit 7 is the y parity, stream bits 8.. are the x coordinate.

use crate::field::Fe;

pub const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// Characters allowed in the 6th position for each y parity.
pub const EVEN_CHARS: &str = "gf2t";
pub const ODD_CHARS: &str = "vdw0";

/// Chars after the 6th one must keep every masked bit inside x's 256 bits.
pub const MAX_TAIL_CHARS: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Network {
    Mainnet,
    Testnet,
}

impl Network {
    pub fn hrp_str(self) -> &'static str {
        match self {
            Network::Mainnet => "sp",
            Network::Testnet => "tsp",
        }
    }
}

fn charset_index(c: u8) -> Option<u8> {
    CHARSET.iter().position(|&x| x == c).map(|i| i as u8)
}

/// A pattern compiled to a mask over the x coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    /// Normalised full text, e.g. `sp1qq?pas` (lowercase, always full form).
    pub text: String,
    /// Prefix chars after `<hrp>1q` (starting with the constant `q`).
    pub chars: Vec<u8>,
    mask: [u8; 32],
    value: [u8; 32],
    /// Number of leading x bytes that carry any masked bit.
    len_bytes: usize,
    /// Mask/value of the first 8 x bytes as a big-endian `u64` (fast path).
    mask_hi: u64,
    value_hi: u64,
    /// Required y parity (`true` = odd) when the 6th char is fixed.
    pub parity: Option<bool>,
    /// Number of constrained x bits (difficulty = 2^bits).
    pub bits: u32,
}

impl Pattern {
    /// Parses either a full pattern (`sp1qq?pas`) or a bare one (`pas`).
    pub fn parse(input: &str, network: Network) -> Result<Pattern, String> {
        let lower = input.trim().to_ascii_lowercase();
        if lower.is_empty() {
            return Err("pattern is empty".to_string());
        }
        let hrp = network.hrp_str();
        let full = format!("{hrp}1qq?");
        let tail: String = if let Some(pos) = lower.find('1') {
            let (given_hrp, rest) = (&lower[..pos], &lower[pos + 1..]);
            if given_hrp != hrp {
                let hint = match given_hrp {
                    "sp" => " (use -n mainnet)",
                    "tsp" => " (use -n testnet)",
                    _ => "",
                };
                return Err(format!(
                    "pattern hrp '{given_hrp}' does not match network hrp '{hrp}'{hint}; \
                     expected a prefix like {full}..."
                ));
            }
            let mut rest_chars = rest.chars();
            match rest_chars.next() {
                None => return Self::compile(network, b"q"),
                Some('q') => {}
                Some(c) => {
                    return Err(format!(
                        "all v0 silent payment addresses start with {hrp}1qq (the char after '1' \
                         is the version, always q), got '{c}' — try {full}{}",
                        &rest[c.len_utf8()..]
                    ));
                }
            }
            match rest_chars.next() {
                None => return Self::compile(network, b"q"),
                Some('q') => {}
                Some(c) => {
                    return Err(format!(
                        "all v0 silent payment addresses start with {hrp}1qq (the char after \
                         {hrp}1q is always q because a compressed key starts with 0x02/0x03), \
                         got '{c}' — try {full}{}",
                        &rest[1..]
                    ));
                }
            }
            rest[2..].to_string()
        } else {
            format!("?{lower}")
        };
        // `tail` = chars from the 6th one on.
        if let Some(c) = tail.chars().next()
            && c != '?'
            && !EVEN_CHARS.contains(c)
            && !ODD_CHARS.contains(c)
        {
            return Err(format!(
                "the 6th char of a silent payment address can only be one of {} (even y) or {} \
                 (odd y), got '{c}' — try {full}{tail}",
                spaced(EVEN_CHARS),
                spaced(ODD_CHARS),
            ));
        }
        let count = tail.chars().count();
        if count > MAX_TAIL_CHARS + 1 {
            return Err(format!(
                "pattern too long: at most {MAX_TAIL_CHARS} chars after the 6th one ({} given)",
                count - 1
            ));
        }
        for c in tail.chars() {
            if c != '?' && !(c.is_ascii() && charset_index(c as u8).is_some()) {
                let note = if "1bio".contains(c) {
                    " (1, b, i and o never appear in a bech32 address)"
                } else {
                    ""
                };
                return Err(format!(
                    "'{c}' is not a bech32 character{note}; charset: {}",
                    String::from_utf8_lossy(CHARSET)
                ));
            }
        }
        let tail = tail.as_bytes();
        let mut chars = Vec::with_capacity(tail.len() + 1);
        chars.push(b'q');
        chars.extend_from_slice(tail);
        Self::compile(network, &chars)
    }

    /// `chars[0]` is the constant `q` group, `chars[1]` the parity group, etc.
    fn compile(network: Network, chars: &[u8]) -> Result<Pattern, String> {
        let mut mask = [0u8; 32];
        let mut value = [0u8; 32];
        let mut parity = None;
        let mut bits = 0u32;
        for (g, &c) in chars.iter().enumerate() {
            if c == b'?' {
                continue;
            }
            let v = charset_index(c).ok_or_else(|| format!("invalid char '{}'", c as char))?;
            for k in 0..5u32 {
                let stream_bit = 5 * g as u32 + k;
                let bit = (v >> (4 - k)) & 1 == 1;
                match stream_bit {
                    0..=5 => {
                        if bit {
                            return Err("internal: constant tag bit set".to_string());
                        }
                    }
                    6 => {
                        if !bit {
                            return Err("internal: constant tag bit clear".to_string());
                        }
                    }
                    7 => parity = Some(bit),
                    _ => {
                        let x_bit = stream_bit - 8;
                        if x_bit >= 256 {
                            return Err("pattern too long".to_string());
                        }
                        let byte = (x_bit / 8) as usize;
                        let m = 0x80u8 >> (x_bit % 8);
                        mask[byte] |= m;
                        if bit {
                            value[byte] |= m;
                        }
                        bits += 1;
                    }
                }
            }
        }
        let len_bytes = mask.iter().rposition(|&m| m != 0).map_or(0, |i| i + 1);
        let mut hi = [0u8; 8];
        hi.copy_from_slice(&mask[..8]);
        let mask_hi = u64::from_be_bytes(hi);
        hi.copy_from_slice(&value[..8]);
        let value_hi = u64::from_be_bytes(hi);
        let text = format!("{}1q{}", network.hrp_str(), String::from_utf8_lossy(chars));
        Ok(Pattern {
            text,
            chars: chars.to_vec(),
            mask,
            value,
            len_bytes,
            mask_hi,
            value_hi,
            parity,
            bits,
        })
    }

    /// Expected number of x candidates to test before a match.
    pub fn expected_candidates(&self) -> f64 {
        2f64.powi(self.bits as i32)
    }

    /// Full match on a canonical x coordinate.
    #[inline]
    pub fn matches_fe(&self, x: &Fe) -> bool {
        if x.top_limb() & self.mask_hi != self.value_hi {
            return false;
        }
        if self.len_bytes <= 8 {
            return true;
        }
        let bytes = x.to_bytes_be();
        self.matches_bytes(&bytes)
    }

    /// Match on a big-endian x coordinate.
    pub fn matches_bytes(&self, x: &[u8; 32]) -> bool {
        (0..self.len_bytes).all(|i| x[i] & self.mask[i] == self.value[i])
    }

    /// Char-level check on a rendered address (independent of the bit layout).
    pub fn matches_address(&self, address: &str) -> bool {
        let text = self.text.as_bytes();
        let addr = address.as_bytes();
        addr.len() >= text.len() && text.iter().zip(addr).all(|(&p, &a)| p == b'?' || p == a)
    }
}

fn spaced(s: &str) -> String {
    s.chars()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Any-of over several patterns.
#[derive(Clone, Debug)]
pub struct PatternSet {
    pub patterns: Vec<Pattern>,
    /// `(mask_hi, value_hi)` of every pattern: the fast reject of the search loop.
    pub keys: Vec<(u64, u64)>,
}

impl PatternSet {
    pub fn parse(inputs: &[String], network: Network) -> Result<PatternSet, String> {
        let patterns = inputs
            .iter()
            .map(|s| Pattern::parse(s, network))
            .collect::<Result<Vec<_>, _>>()?;
        let keys = patterns.iter().map(|p| (p.mask_hi, p.value_hi)).collect();
        Ok(PatternSet { patterns, keys })
    }

    /// Index of the first matching pattern for this x coordinate.
    #[inline]
    pub fn find(&self, x: &Fe) -> Option<usize> {
        self.patterns.iter().position(|p| p.matches_fe(x))
    }

    /// Expected candidates for a hit on any pattern (union approximated as sum of rates).
    pub fn expected_candidates(&self) -> f64 {
        let rate: f64 = self
            .patterns
            .iter()
            .map(|p| 1.0 / p.expected_candidates())
            .sum();
        1.0 / rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address;
    use k256::elliptic_curve::PrimeField;
    use k256::elliptic_curve::sec1::ToSec1Point;
    use k256::{ProjectivePoint, Scalar};

    #[test]
    fn bare_form_expands_to_full() {
        let p = Pattern::parse("pas", Network::Mainnet).unwrap();
        assert_eq!(p.text, "sp1qq?pas");
        assert_eq!(p.parity, None);
        assert_eq!(p.bits, 15);
        let t = Pattern::parse("pas", Network::Testnet).unwrap();
        assert_eq!(t.text, "tsp1qq?pas");
        assert_eq!(Pattern::parse("PAS", Network::Mainnet).unwrap(), p);
    }

    #[test]
    fn full_form_with_parity() {
        let even = Pattern::parse("sp1qqgpas", Network::Mainnet).unwrap();
        assert_eq!(even.parity, Some(false));
        assert_eq!(even.bits, 17);
        assert_eq!(even.expected_candidates(), 131072.0);
        let odd = Pattern::parse("sp1qqvpas", Network::Mainnet).unwrap();
        assert_eq!(odd.parity, Some(true));
        let wild = Pattern::parse("sp1qq?pas", Network::Mainnet).unwrap();
        assert_eq!(wild, Pattern::parse("pas", Network::Mainnet).unwrap());
        let t = Pattern::parse("tsp1qqgpas", Network::Testnet).unwrap();
        assert_eq!(t.text, "tsp1qqgpas");
    }

    #[test]
    fn empty_tail_matches_everything() {
        let p = Pattern::parse("sp1qq", Network::Mainnet).unwrap();
        assert_eq!(p.bits, 0);
        assert!(p.matches_fe(&Fe::ZERO));
        let p = Pattern::parse("sp1q", Network::Mainnet).unwrap();
        assert_eq!(p.bits, 0);
    }

    #[test]
    fn error_fifth_char() {
        let err = Pattern::parse("sp1qpas", Network::Mainnet).unwrap_err();
        assert!(err.contains("sp1qq"), "{err}");
        assert!(err.contains("try sp1qq?pas"), "{err}");
    }

    #[test]
    fn error_version_char() {
        let err = Pattern::parse("sp1ppas", Network::Mainnet).unwrap_err();
        assert!(err.contains("version"), "{err}");
    }

    #[test]
    fn error_sixth_char() {
        let err = Pattern::parse("sp1qqpas", Network::Mainnet).unwrap_err();
        assert!(err.contains("g f 2 t"), "{err}");
        assert!(err.contains("v d w 0"), "{err}");
        assert!(err.contains("try sp1qq?pas"), "{err}");
    }

    #[test]
    fn error_invalid_charset() {
        let err = Pattern::parse("pasb", Network::Mainnet).unwrap_err();
        assert!(err.contains("'b' is not a bech32 character"), "{err}");
        assert!(err.contains("never appear"), "{err}");
        let err = Pattern::parse("ln#", Network::Mainnet).unwrap_err();
        assert!(err.contains("'#' is not a bech32 character"), "{err}");
        let err = Pattern::parse("pastá", Network::Mainnet).unwrap_err();
        assert!(err.contains("'á' is not a bech32 character"), "{err}");
        let err = Pattern::parse("sp1qqépas", Network::Mainnet).unwrap_err();
        assert!(err.contains("got 'é'"), "{err}");
        assert!(err.contains("try sp1qq?épas"), "{err}");
        let err = Pattern::parse("sp1qépas", Network::Mainnet).unwrap_err();
        assert!(err.contains("got 'é'"), "{err}");
        let err = Pattern::parse("sp1épas", Network::Mainnet).unwrap_err();
        assert!(err.contains("got 'é'"), "{err}");
    }

    #[test]
    fn error_wrong_hrp() {
        let err = Pattern::parse("tsp1qq?pas", Network::Mainnet).unwrap_err();
        assert!(err.contains("-n testnet"), "{err}");
        let err = Pattern::parse("sp1qq?pas", Network::Testnet).unwrap_err();
        assert!(err.contains("-n mainnet"), "{err}");
        let err = Pattern::parse("bc1qq?pas", Network::Mainnet).unwrap_err();
        assert!(err.contains("does not match"), "{err}");
    }

    #[test]
    fn error_too_long() {
        let ok = "q".repeat(MAX_TAIL_CHARS);
        let p = Pattern::parse(&format!("sp1qq?{ok}"), Network::Mainnet).unwrap();
        assert_eq!(p.bits, 250);
        let too_long = "q".repeat(MAX_TAIL_CHARS + 1);
        let err = Pattern::parse(&format!("sp1qq?{too_long}"), Network::Mainnet).unwrap_err();
        assert!(err.contains("too long"), "{err}");
        let err = Pattern::parse("", Network::Mainnet).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn mask_layout_matches_encoder() {
        // For random keys, the compiled mask must agree with the real address
        // for every prefix length, with and without the parity char.
        let spend = [2u8; 33];
        let mut seed = 0x1234_5678u64;
        for _ in 0..40 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut bytes = [0u8; 32];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (seed >> (i % 8 * 8)) as u8 ^ (i as u8).wrapping_mul(31);
            }
            let scalar = Scalar::from_repr_vartime(bytes.into()).unwrap();
            let affine = (ProjectivePoint::GENERATOR * scalar).to_affine();
            let sec1 = affine.to_sec1_point(true);
            let scan: [u8; 33] = sec1.as_bytes().try_into().unwrap();
            let x: [u8; 32] = scan[1..].try_into().unwrap();
            let odd = scan[0] == 0x03;
            let addr = address::encode(address::HRP_MAINNET, &scan, &spend);
            for len in 5..=56 {
                let prefix = &addr[..len];
                let fixed = Pattern::parse(prefix, Network::Mainnet).unwrap();
                assert!(fixed.matches_bytes(&x), "{prefix} vs {addr}");
                assert!(fixed.matches_address(&addr));
                if len >= 6 {
                    assert_eq!(fixed.parity, Some(odd));
                    let mut wild = prefix.to_string();
                    wild.replace_range(5..6, "?");
                    let w = Pattern::parse(&wild, Network::Mainnet).unwrap();
                    assert!(w.matches_bytes(&x));
                    assert_eq!(w.parity, None);
                    assert_eq!(w.bits + 2, fixed.bits);
                }
                // A flipped last char must not match.
                if len >= 7 {
                    let last = prefix.as_bytes()[len - 1];
                    let other = CHARSET[(charset_index(last).unwrap() as usize + 1) % 32];
                    let mut flipped = prefix.as_bytes().to_vec();
                    flipped[len - 1] = other;
                    let f = Pattern::parse(&String::from_utf8(flipped).unwrap(), Network::Mainnet)
                        .unwrap();
                    assert!(!f.matches_bytes(&x), "{prefix} flipped matched");
                    assert!(!f.matches_address(&addr));
                }
            }
            // Also exercise the Fe fast path.
            let fe = Fe::from_bytes_be(&x);
            let long = Pattern::parse(&addr[..30], Network::Mainnet).unwrap();
            assert!(long.matches_fe(&fe));
            let short = Pattern::parse(&addr[..9], Network::Mainnet).unwrap();
            assert!(short.matches_fe(&fe));
        }
    }

    #[test]
    fn pattern_set_any_of() {
        let set = PatternSet::parse(
            &["pas".to_string(), "sp1qqgacd".to_string()],
            Network::Mainnet,
        )
        .unwrap();
        assert_eq!(set.patterns.len(), 2);
        let e = set.expected_candidates();
        assert!(e > 2f64.powi(14) && e < 2f64.powi(15), "{e}");
    }
}
