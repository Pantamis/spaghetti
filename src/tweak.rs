//! Split-key tweak `<t>/<e>/<s>`: the public data that turns a BIP32-derived
//! scan key `d` into the vanity scan key `s·λ^e·(d + t) mod n`.
//!
//! Range arithmetic shared by the search and `recover`: split-mode worker `i`
//! walks offsets `t ∈ [i·2^THREAD_RANGE_BITS, (i+1)·2^THREAD_RANGE_BITS)`. With
//! at most `2^(MAX_TWEAK_BITS − THREAD_RANGE_BITS) = 256` workers every visited
//! `t` is below `2^MAX_TWEAK_BITS`, which is the range `recover` scans.

use std::fmt;
use std::str::FromStr;

use k256::{ProjectivePoint, Scalar};

use crate::search::lambda;

/// Offsets covered by one split-mode worker: `2^44`.
pub const THREAD_RANGE_BITS: u32 = 44;
/// Upper bound on every published tweak offset: `2^52`.
pub const MAX_TWEAK_BITS: u32 = 52;
/// Maximum number of split-mode workers, `2^(52 − 44)`.
pub const MAX_SPLIT_THREADS: usize = 1 << (MAX_TWEAK_BITS - THREAD_RANGE_BITS);

/// `λ^e` for `e ∈ {0, 1, 2}` (`λ³ = 1`, so any `e` is reduced mod 3).
pub fn lambda_pow(e: u8) -> Scalar {
    let mut out = Scalar::ONE;
    for _ in 0..e % 3 {
        out = out.mul(&lambda());
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tweak {
    /// Additive offset, `< 2^MAX_TWEAK_BITS`.
    pub t: u64,
    /// Endomorphism power `e ∈ 0..=2`.
    pub endo: u8,
    /// `s = -1` when true (parity fix), `+1` otherwise.
    pub negate: bool,
}

impl Tweak {
    /// `s·λ^e·(d + t) mod n`.
    pub fn apply(&self, d: &Scalar) -> Scalar {
        let k = d.add(&Scalar::from(self.t)).mul(&lambda_pow(self.endo));
        if self.negate { k.negate() } else { k }
    }

    /// `s·λ^e·(D + t·G)`.
    pub fn apply_point(&self, base: &ProjectivePoint) -> ProjectivePoint {
        let p = (*base + ProjectivePoint::GENERATOR * Scalar::from(self.t)) * lambda_pow(self.endo);
        if self.negate { -p } else { p }
    }

    /// Human formula, e.g. `-λ^1·(d + 42) mod n`.
    pub fn formula(&self) -> String {
        let sign = if self.negate { "-" } else { "" };
        format!("{sign}λ^{}·(d + {}) mod n", self.endo, self.t)
    }
}

impl fmt::Display for Tweak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.negate { '-' } else { '+' };
        write!(f, "{}/{}/{sign}", self.t, self.endo)
    }
}

impl FromStr for Tweak {
    type Err = String;

    fn from_str(text: &str) -> Result<Tweak, String> {
        let bad = || format!("invalid tweak '{text}': expected <t>/<e>/<s> like 48213946821/1/-");
        let mut parts = text.trim().split('/');
        let (t, e, s) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(t), Some(e), Some(s), None) => (t, e, s),
            _ => return Err(bad()),
        };
        if t.is_empty() || !t.bytes().all(|c| c.is_ascii_digit()) {
            return Err(bad());
        }
        let t: u64 = t.parse().map_err(|_| bad())?;
        if t >= 1u64 << MAX_TWEAK_BITS {
            return Err(format!(
                "invalid tweak '{text}': offset must be below 2^{MAX_TWEAK_BITS}"
            ));
        }
        let endo = match e {
            "0" => 0,
            "1" => 1,
            "2" => 2,
            _ => return Err(bad()),
        };
        let negate = match s {
            "+" => false,
            "-" => true,
            _ => return Err(bad()),
        };
        Ok(Tweak { t, endo, negate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::NonZeroScalar;
    use k256::elliptic_curve::Generate;

    fn random_scalar() -> Scalar {
        *NonZeroScalar::try_generate().unwrap().as_ref()
    }

    pub fn variants(t: u64) -> Vec<Tweak> {
        let mut out = Vec::new();
        for endo in 0..3 {
            for negate in [false, true] {
                out.push(Tweak { t, endo, negate });
            }
        }
        out
    }

    #[test]
    fn display_and_parse_roundtrip() {
        let tweak: Tweak = "48213946821/1/-".parse().unwrap();
        assert_eq!(
            tweak,
            Tweak {
                t: 48213946821,
                endo: 1,
                negate: true
            }
        );
        assert_eq!(tweak.to_string(), "48213946821/1/-");
        for tweak in variants(7).into_iter().chain(variants(0)) {
            assert_eq!(tweak.to_string().parse::<Tweak>().unwrap(), tweak);
        }
        let max = Tweak {
            t: (1 << MAX_TWEAK_BITS) - 1,
            endo: 2,
            negate: false,
        };
        assert_eq!(max.to_string().parse::<Tweak>().unwrap(), max);
        assert_eq!(" 5/0/+ ".parse::<Tweak>().unwrap().t, 5);
    }

    #[test]
    fn parse_rejects_malformed() {
        for bad in [
            "",
            "5",
            "5/0",
            "5/0/+/",
            "5/3/+",
            "5/0/*",
            "-5/0/+",
            "5_000/0/+",
            "4503599627370496/0/+",
            "x/0/+",
            "5/0/",
        ] {
            assert!(bad.parse::<Tweak>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn lambda_powers() {
        assert_eq!(lambda_pow(0), Scalar::ONE);
        assert_eq!(lambda_pow(1), lambda());
        assert_eq!(lambda_pow(2), lambda().mul(&lambda()));
        assert_eq!(lambda_pow(2).mul(&lambda()), Scalar::ONE);
        assert_eq!(lambda_pow(3), Scalar::ONE);
    }

    #[test]
    fn apply_matches_apply_point() {
        let d = random_scalar();
        let base = ProjectivePoint::GENERATOR * d;
        for tweak in variants(48213946821).into_iter().chain(variants(0)) {
            let k = tweak.apply(&d);
            assert_eq!(
                ProjectivePoint::GENERATOR * k,
                tweak.apply_point(&base),
                "{tweak}"
            );
        }
        // The six variants of one offset are six distinct points.
        let points: Vec<_> = variants(1)
            .iter()
            .map(|t| t.apply_point(&base).to_affine())
            .collect();
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn formula_text() {
        let tweak: Tweak = "42/1/-".parse().unwrap();
        assert_eq!(tweak.formula(), "-λ^1·(d + 42) mod n");
        let tweak: Tweak = "42/0/+".parse().unwrap();
        assert_eq!(tweak.formula(), "λ^0·(d + 42) mod n");
    }
}
