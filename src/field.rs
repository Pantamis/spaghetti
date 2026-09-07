//! secp256k1 base field arithmetic, p = 2^256 - 2^32 - 977.
//!
//! Elements are four little-endian `u64` limbs and are kept fully reduced
//! (canonical, `< p`) after every operation. Products are reduced with two
//! folds of the high half times `c = 2^32 + 977` (since `2^256 ≡ c mod p`),
//! followed by a single conditional subtraction of `p`.

/// `2^256 mod p`.
const C: u64 = 0x1_0000_03D1;

const P: [u64; 4] = [
    0xFFFF_FFFE_FFFF_FC2F,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// A canonical secp256k1 field element.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fe(pub [u64; 4]);

impl Fe {
    pub const ZERO: Fe = Fe([0, 0, 0, 0]);
    #[cfg_attr(not(test), allow(dead_code))]
    pub const ONE: Fe = Fe([1, 0, 0, 0]);

    /// Cube root of unity β: λ·(x, y) = (β·x, y).
    pub const BETA: Fe = Fe([
        0xc139_6c28_7195_01ee,
        0x9cf0_4975_12f5_8995,
        0x6e64_479e_ac34_34e9,
        0x7ae9_6a2b_657c_0710,
    ]);

    /// β² (the other non-trivial cube root of unity): λ²·(x, y) = (β²·x, y).
    pub const BETA2: Fe = Fe([
        0x3ec6_93d6_8e6a_fa40,
        0x630f_b68a_ed0a_766a,
        0x919b_b861_53cb_cb16,
        0x8516_95d4_9a83_f8ef,
    ]);

    /// Parses a big-endian 32-byte integer; values `>= p` are reduced once.
    pub fn from_bytes_be(bytes: &[u8; 32]) -> Fe {
        let limb = |i: usize| -> u64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            u64::from_be_bytes(b)
        };
        let mut r = Fe([limb(3), limb(2), limb(1), limb(0)]);
        r.reduce_once();
        r
    }

    pub fn to_bytes_be(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.0[3 - i].to_be_bytes());
        }
        out
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    #[inline(always)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    /// Big-endian-most limb, i.e. the first 8 bytes of `to_bytes_be` as a `u64`.
    #[inline(always)]
    pub fn top_limb(&self) -> u64 {
        self.0[3]
    }

    #[inline(always)]
    fn geq_p(&self) -> bool {
        let a = &self.0;
        if a[3] != P[3] || a[2] != P[2] || a[1] != P[1] {
            // Since P[1..4] are all ones, any difference means a < p.
            return false;
        }
        a[0] >= P[0]
    }

    /// Subtracts `p` if the value is `>= p` (input must be `< 2p`).
    #[inline(always)]
    fn reduce_once(&mut self) {
        if self.geq_p() {
            self.sub_p_wrapping();
        }
    }

    #[inline(always)]
    fn sub_p_wrapping(&mut self) {
        let a = &mut self.0;
        let (r0, b0) = a[0].overflowing_sub(P[0]);
        let (r1, b1) = sbb(a[1], P[1], b0);
        let (r2, b2) = sbb(a[2], P[2], b1);
        let (r3, _) = sbb(a[3], P[3], b2);
        *a = [r0, r1, r2, r3];
    }

    #[inline(always)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn add(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let (r0, c0) = a[0].overflowing_add(b[0]);
        let (r1, c1) = adc(a[1], b[1], c0);
        let (r2, c2) = adc(a[2], b[2], c1);
        let (r3, c3) = adc(a[3], b[3], c2);
        let mut r = Fe([r0, r1, r2, r3]);
        // a + b < 2p < 2^257: a carry means the true value is 2^256 + low,
        // and (2^256 + low) - p fits in 256 bits, so the wrapping subtraction is exact.
        if c3 || r.geq_p() {
            r.sub_p_wrapping();
        }
        r
    }

    #[inline(always)]
    pub fn sub(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let (r0, b0) = a[0].overflowing_sub(b[0]);
        let (r1, b1) = sbb(a[1], b[1], b0);
        let (r2, b2) = sbb(a[2], b[2], b1);
        let (r3, b3) = sbb(a[3], b[3], b2);
        let mut r = [r0, r1, r2, r3];
        if b3 {
            // Wrapped below zero: add p back (r + p wraps to the right value).
            let (s0, c0) = r[0].overflowing_add(P[0]);
            let (s1, c1) = adc(r[1], P[1], c0);
            let (s2, c2) = adc(r[2], P[2], c1);
            let (s3, _) = adc(r[3], P[3], c2);
            r = [s0, s1, s2, s3];
        }
        Fe(r)
    }

    #[inline(always)]
    pub fn neg(&self) -> Fe {
        if self.is_zero() {
            return Fe::ZERO;
        }
        Fe(P).sub(self)
    }

    #[inline(always)]
    pub fn mul(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let mut w = [0u64; 8];
        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..4 {
                let (lo, hi) = mac(w[i + j], a[i], b[j], carry);
                w[i + j] = lo;
                carry = hi;
            }
            w[i + 4] = carry;
        }
        reduce_wide(&w)
    }

    #[inline(always)]
    pub fn square(&self) -> Fe {
        let a = &self.0;
        // Off-diagonal products, each counted once, then doubled.
        let mut w = [0u64; 8];
        let mut carry = 0u64;
        for j in 1..4 {
            let (lo, hi) = mac(0, a[0], a[j], carry);
            w[j] = lo;
            carry = hi;
        }
        w[4] = carry;
        carry = 0;
        for j in 2..4 {
            let (lo, hi) = mac(w[1 + j], a[1], a[j], carry);
            w[1 + j] = lo;
            carry = hi;
        }
        w[5] = carry;
        let (lo, hi) = mac(w[5], a[2], a[3], 0);
        w[5] = lo;
        w[6] = hi;
        // Double the off-diagonal sum.
        let mut c = 0u64;
        for limb in w.iter_mut().take(7).skip(1) {
            let t = ((*limb as u128) << 1) | (c as u128);
            *limb = t as u64;
            c = (t >> 64) as u64;
        }
        w[7] = c;
        // Add the squares on the diagonal.
        let mut carry = 0u64;
        for i in 0..4 {
            let sq = (a[i] as u128) * (a[i] as u128);
            let t0 = (w[2 * i] as u128) + (sq as u64 as u128) + (carry as u128);
            w[2 * i] = t0 as u64;
            let t1 = (w[2 * i + 1] as u128) + (sq >> 64) + (t0 >> 64);
            w[2 * i + 1] = t1 as u64;
            carry = (t1 >> 64) as u64;
        }
        reduce_wide(&w)
    }

    /// Multiplicative inverse via Fermat (`a^(p-2)`), using libsecp256k1's
    /// addition chain. `invert(0) == 0`.
    pub fn invert(&self) -> Fe {
        let x = *self;
        let x2 = x.square().mul(&x);
        let x3 = x2.square().mul(&x);
        let x6 = x3.sqn(3).mul(&x3);
        let x9 = x6.sqn(3).mul(&x3);
        let x11 = x9.sqn(2).mul(&x2);
        let x22 = x11.sqn(11).mul(&x11);
        let x44 = x22.sqn(22).mul(&x22);
        let x88 = x44.sqn(44).mul(&x44);
        let x176 = x88.sqn(88).mul(&x88);
        let x220 = x176.sqn(44).mul(&x44);
        let x223 = x220.sqn(3).mul(&x3);
        let t = x223.sqn(23).mul(&x22);
        let t = t.sqn(5).mul(&x);
        let t = t.sqn(3).mul(&x2);
        t.sqn(2).mul(&x)
    }

    fn sqn(&self, n: u32) -> Fe {
        let mut r = *self;
        for _ in 0..n {
            r = r.square();
        }
        r
    }
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (s, c1) = a.overflowing_add(b);
    let (s, c2) = s.overflowing_add(u64::from(carry));
    (s, c1 | c2)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: bool) -> (u64, bool) {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(u64::from(borrow));
    (d, b1 | b2)
}

/// `acc + a * b + carry` as (lo, hi).
#[inline(always)]
fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (acc as u128) + (a as u128) * (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

/// Reduces a 512-bit little-endian product to a canonical element.
#[inline]
fn reduce_wide(w: &[u64; 8]) -> Fe {
    // First fold: r = lo + hi * c. hi*c < 2^289, so the carry out is < 2^34.
    let mut r = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (lo, hi) = mac(w[i], w[i + 4], C, carry);
        r[i] = lo;
        carry = hi;
    }
    // Second fold: r += carry * c (< 2^67).
    let t = (r[0] as u128) + (carry as u128) * (C as u128);
    r[0] = t as u64;
    let mut c = (t >> 64) as u64;
    for limb in r.iter_mut().skip(1) {
        let t = (*limb as u128) + (c as u128);
        *limb = t as u64;
        c = (t >> 64) as u64;
    }
    if c != 0 {
        // Third fold: the value was 2^256 + tiny, so add c once more (cannot overflow).
        let (lo, c0) = r[0].overflowing_add(C);
        r[0] = lo;
        let (l1, c1) = adc(r[1], 0, c0);
        r[1] = l1;
        let (l2, c2) = adc(r[2], 0, c1);
        r[2] = l2;
        r[3] = r[3].wrapping_add(u64::from(c2));
    }
    let mut fe = Fe(r);
    fe.reduce_once();
    fe
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use num_traits::{One, Zero};

    fn p() -> BigUint {
        (BigUint::one() << 256u32) - (BigUint::one() << 32u32) - BigUint::from(977u32)
    }

    fn to_big(fe: &Fe) -> BigUint {
        BigUint::from_bytes_be(&fe.to_bytes_be())
    }

    fn from_big(v: &BigUint) -> Fe {
        let bytes = (v % p()).to_bytes_be();
        let mut buf = [0u8; 32];
        buf[32 - bytes.len()..].copy_from_slice(&bytes);
        Fe::from_bytes_be(&buf)
    }

    /// Deterministic xorshift so the tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn fe(&mut self) -> Fe {
            let mut limbs = [0u64; 4];
            for l in &mut limbs {
                *l = self.next();
            }
            // Bias toward high values (>= 2^255) and near-p values now and then.
            match self.next() % 8 {
                0 => limbs[3] |= 1 << 63,
                1 => {
                    limbs = P;
                    limbs[0] = limbs[0].wrapping_sub(self.next() % 4096);
                }
                _ => {}
            }
            let mut fe = Fe(limbs);
            fe.reduce_once();
            fe
        }
    }

    fn edge_cases() -> Vec<Fe> {
        let pm1 = from_big(&(p() - BigUint::one()));
        vec![
            Fe::ZERO,
            Fe::ONE,
            pm1,
            Fe([0, 0, 0, 1 << 63]),
            Fe([u64::MAX, u64::MAX, u64::MAX, 1 << 63]),
            Fe::BETA,
            Fe::BETA2,
            Fe([0xFFFF_FFFE_FFFF_FC2E, u64::MAX, u64::MAX, u64::MAX]),
            Fe([0x1_0000_03D1, 0, 0, 0]),
        ]
    }

    fn check_canonical(fe: &Fe) {
        assert!(to_big(fe) < p(), "not canonical: {fe:?}");
    }

    #[test]
    fn from_bytes_reduces_values_at_or_above_p() {
        let pb = p().to_bytes_be();
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&pb);
        assert_eq!(Fe::from_bytes_be(&buf), Fe::ZERO);
        buf = [0xFF; 32];
        let expect = from_big(&((BigUint::one() << 256u32) - BigUint::one()));
        assert_eq!(Fe::from_bytes_be(&buf), expect);
        check_canonical(&expect);
    }

    #[test]
    fn bytes_roundtrip() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..1000 {
            let a = rng.fe();
            assert_eq!(Fe::from_bytes_be(&a.to_bytes_be()), a);
        }
    }

    #[test]
    fn ops_match_bigint_reference() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let pp = p();
        let mut pairs: Vec<(Fe, Fe)> = Vec::new();
        for a in edge_cases() {
            for b in edge_cases() {
                pairs.push((a, b));
            }
        }
        for _ in 0..10_000 {
            pairs.push((rng.fe(), rng.fe()));
        }
        for (a, b) in &pairs {
            let (ba, bb) = (to_big(a), to_big(b));
            let sum = a.add(b);
            check_canonical(&sum);
            assert_eq!(to_big(&sum), (&ba + &bb) % &pp, "add {a:?} {b:?}");

            let diff = a.sub(b);
            check_canonical(&diff);
            assert_eq!(to_big(&diff), (&ba + &pp - &bb) % &pp, "sub {a:?} {b:?}");

            let prod = a.mul(b);
            check_canonical(&prod);
            assert_eq!(to_big(&prod), (&ba * &bb) % &pp, "mul {a:?} {b:?}");

            let sq = a.square();
            check_canonical(&sq);
            assert_eq!(to_big(&sq), (&ba * &ba) % &pp, "square {a:?}");
            assert_eq!(sq, a.mul(a));

            let neg = a.neg();
            check_canonical(&neg);
            assert_eq!(to_big(&neg), (&pp - &ba) % &pp, "neg {a:?}");
            assert_eq!(a.add(&neg), Fe::ZERO);

            assert_eq!(a.is_zero(), ba.is_zero());
            assert_eq!(a.is_odd(), (&ba & BigUint::one()) == BigUint::one());
        }
    }

    #[test]
    fn invert_matches_reference_and_roundtrips() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let pp = p();
        let exp = &pp - BigUint::from(2u32);
        let mut values = edge_cases();
        for _ in 0..500 {
            values.push(rng.fe());
        }
        for a in &values {
            let inv = a.invert();
            check_canonical(&inv);
            assert_eq!(to_big(&inv), to_big(a).modpow(&exp, &pp), "invert {a:?}");
            if a.is_zero() {
                assert_eq!(inv, Fe::ZERO);
            } else {
                assert_eq!(a.mul(&inv), Fe::ONE, "a * a^-1 {a:?}");
            }
        }
    }

    #[test]
    fn beta_constants() {
        assert_eq!(Fe::BETA.square(), Fe::BETA2);
        assert_eq!(Fe::BETA.mul(&Fe::BETA2), Fe::ONE);
        assert_eq!(Fe::BETA.square().mul(&Fe::BETA), Fe::ONE);
        assert_ne!(Fe::BETA, Fe::ONE);
        assert_ne!(Fe::BETA2, Fe::ONE);
        let beta_hex = "7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee";
        assert_eq!(hex::encode(Fe::BETA.to_bytes_be()), beta_hex);
    }

    #[test]
    fn top_limb_is_first_eight_bytes() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for _ in 0..100 {
            let a = rng.fe();
            let bytes = a.to_bytes_be();
            let mut head = [0u8; 8];
            head.copy_from_slice(&bytes[..8]);
            assert_eq!(a.top_limb(), u64::from_be_bytes(head));
        }
    }
}
