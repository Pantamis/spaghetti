//! secp256k1 base field arithmetic, p = 2^256 - 2^32 - 977.
//!
//! Elements are four little-endian `u64` limbs and are kept fully reduced
//! (canonical, `< p`) after every operation. Products are reduced with two
//! folds of the high half times `c = 2^32 + 977` (since `2^256 ≡ c mod p`),
//! followed by a single conditional subtraction of `p`.
//!
//! The hot operations have an opt-in AArch64 assembly version
//! (`field/aarch64.rs`, feature `asm`, the crate's only `unsafe`); the
//! portable Rust version is the default everywhere and the reference the
//! assembly is tested against.

#[cfg(all(target_arch = "aarch64", feature = "asm"))]
mod aarch64;

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
pub struct Fe([u64; 4]);

impl Fe {
    pub const ZERO: Fe = Fe([0, 0, 0, 0]);
    pub const ONE: Fe = Fe([1, 0, 0, 0]);

    /// Cube root of unity β: λ·(x, y) = (β·x, y).
    pub const BETA: Fe = Fe([
        0xc139_6c28_7195_01ee,
        0x9cf0_4975_12f5_8995,
        0x6e64_479e_ac34_34e9,
        0x7ae9_6a2b_657c_0710,
    ]);

    /// β² (the other non-trivial cube root of unity): λ²·(x, y) = (β²·x, y).
    /// Reference value for the tests; the search uses β² = −β − 1 instead.
    #[cfg(test)]
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
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    /// Big-endian-most limb, i.e. the first 8 bytes of `to_bytes_be` as a `u64`.
    #[inline(always)]
    pub fn top_limb(&self) -> u64 {
        self.0[3]
    }

    /// `self.neg().top_limb()` for a non-zero `self`, without the other limbs:
    /// `p − s` borrows into the top limb exactly when the low 192 bits of `s`
    /// exceed those of `p` (all ones above `P[0]`).
    #[inline(always)]
    pub fn neg_top_limb(&self) -> u64 {
        let s = &self.0;
        let borrow = (s[1] & s[2]) == u64::MAX && s[0] > P[0];
        (!s[3]).wrapping_sub(u64::from(borrow))
    }

    /// Subtracts `p` if the value is `>= p` (input must be `< 2p`).
    /// `r >= p` exactly when `r + c` carries out of 256 bits, and then
    /// `r - p = (r + c) mod 2^256`.
    #[inline(always)]
    fn reduce_once(&mut self) {
        let (t, c) = add_c(&self.0);
        if c {
            self.0 = t;
        }
    }

    #[inline(always)]
    pub fn add(&self, rhs: &Fe) -> Fe {
        #[cfg(all(target_arch = "aarch64", feature = "asm"))]
        {
            Fe(aarch64::add(&self.0, &rhs.0))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm")))]
        {
            self.add_generic(rhs)
        }
    }

    #[inline(always)]
    pub fn sub(&self, rhs: &Fe) -> Fe {
        #[cfg(all(target_arch = "aarch64", feature = "asm"))]
        {
            Fe(aarch64::sub(&self.0, &rhs.0))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm")))]
        {
            self.sub_generic(rhs)
        }
    }

    #[inline(always)]
    pub fn neg(&self) -> Fe {
        if self.is_zero() {
            return zero_cold();
        }
        #[cfg(all(target_arch = "aarch64", feature = "asm"))]
        {
            Fe(aarch64::neg(&self.0))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm")))]
        {
            self.neg_generic()
        }
    }

    #[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
    #[inline(always)]
    pub fn add_generic(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let (r0, c) = a[0].overflowing_add(b[0]);
        let (r1, c) = adc(a[1], b[1], c);
        let (r2, c) = adc(a[2], b[2], c);
        let (r3, c3) = adc(a[3], b[3], c);
        let r = [r0, r1, r2, r3];
        // a + b < 2p: subtract p (add c) once if the sum carried out of 256
        // bits or is still >= p; both cases are caught by the carry of r + c.
        let (t, c) = add_c(&r);
        Fe(select(c3 | c, &t, &r))
    }

    /// Subtraction as a borrow chain written in complement form (see [`sbb`]).
    #[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
    #[inline(always)]
    pub fn sub_generic(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let (r0, borrow) = a[0].overflowing_sub(b[0]);
        let (r1, carry) = sbb(a[1], b[1], !borrow);
        let (r2, carry) = sbb(a[2], b[2], carry);
        let (r3, carry) = sbb(a[3], b[3], carry);
        // Wrapped below zero (no carry out): add p back, i.e. subtract
        // c = 2^256 - p from the wrapped value (which is > c, so this cannot
        // underflow).
        let c = if carry { 0 } else { C };
        let (r0, borrow) = r0.overflowing_sub(c);
        let (r1, carry) = sbb(r1, 0, !borrow);
        let (r2, carry) = sbb(r2, 0, carry);
        let (r3, _) = sbb(r3, 0, carry);
        Fe([r0, r1, r2, r3])
    }

    /// `p − s` for non-zero `s`.
    #[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
    #[inline(always)]
    pub fn neg_generic(&self) -> Fe {
        // 0 < s < p, so p − s never borrows out of 256 bits and is canonical;
        // P[1..4] are all ones.
        let s = &self.0;
        let (r0, borrow) = P[0].overflowing_sub(s[0]);
        let (r1, carry) = sbb(u64::MAX, s[1], !borrow);
        let (r2, carry) = sbb(u64::MAX, s[2], carry);
        let (r3, _) = sbb(u64::MAX, s[3], carry);
        Fe([r0, r1, r2, r3])
    }

    #[inline(always)]
    pub fn mul(&self, rhs: &Fe) -> Fe {
        #[cfg(all(target_arch = "aarch64", feature = "asm"))]
        {
            let (r, rare) = aarch64::mul(&self.0, &rhs.0);
            finish(r, rare)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm")))]
        {
            self.mul_generic(rhs)
        }
    }

    #[inline(always)]
    pub fn square(&self) -> Fe {
        #[cfg(all(target_arch = "aarch64", feature = "asm"))]
        {
            let (r, rare) = aarch64::square(&self.0);
            finish(r, rare)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "asm")))]
        {
            self.square_generic()
        }
    }

    /// Schoolbook product, one row per limb of `self`; each row adds its low
    /// and high halves as two carry chains so the compiler can use the flags.
    #[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
    #[inline(always)]
    pub fn mul_generic(&self, rhs: &Fe) -> Fe {
        let a = &self.0;
        let b = &rhs.0;
        let (w0, h0) = mul_wide(a[0], b[0]);
        let (l1, h1) = mul_wide(a[0], b[1]);
        let (l2, h2) = mul_wide(a[0], b[2]);
        let (l3, h3) = mul_wide(a[0], b[3]);
        let (w1, c) = l1.overflowing_add(h0);
        let (w2, c) = adc(l2, h1, c);
        let (w3, c) = adc(l3, h2, c);
        let w4 = h3 + u64::from(c);
        let mut w = [w0, w1, w2, w3, w4, 0, 0, 0];
        mul_row(&mut w, 1, a[1], b);
        mul_row(&mut w, 2, a[2], b);
        mul_row(&mut w, 3, a[3], b);
        reduce_wide(&w)
    }

    #[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
    #[inline(always)]
    pub fn square_generic(&self) -> Fe {
        let [a0, a1, a2, a3] = self.0;
        // Off-diagonal products, each counted once.
        let (w1, h01) = mul_wide(a0, a1);
        let (l02, h02) = mul_wide(a0, a2);
        let (l03, h03) = mul_wide(a0, a3);
        let (w2, c) = l02.overflowing_add(h01);
        let (w3, c) = adc(l03, h02, c);
        let w4 = h03 + u64::from(c);
        let (l12, h12) = mul_wide(a1, a2);
        let (l13, h13) = mul_wide(a1, a3);
        let (w3, c) = w3.overflowing_add(l12);
        let (w4, c) = adc(w4, l13, c);
        let w5 = u64::from(c);
        let (w4, c) = w4.overflowing_add(h12);
        let w5 = w5 + h13 + u64::from(c);
        let (l23, h23) = mul_wide(a2, a3);
        let (w5, c) = w5.overflowing_add(l23);
        let w6 = h23 + u64::from(c);
        // Double them (the sum is < 2^448, so the top limb is just the shifted-out bit).
        let w7 = w6 >> 63;
        let w6 = (w6 << 1) | (w5 >> 63);
        let w5 = (w5 << 1) | (w4 >> 63);
        let w4 = (w4 << 1) | (w3 >> 63);
        let w3 = (w3 << 1) | (w2 >> 63);
        let w2 = (w2 << 1) | (w1 >> 63);
        let w1 = w1 << 1;
        // Add the squares on the diagonal.
        let (w0, d0) = mul_wide(a0, a0);
        let (l1, d1) = mul_wide(a1, a1);
        let (l2, d2) = mul_wide(a2, a2);
        let (l3, d3) = mul_wide(a3, a3);
        let (w1, c) = w1.overflowing_add(d0);
        let (w2, c) = adc(w2, l1, c);
        let (w3, c) = adc(w3, d1, c);
        let (w4, c) = adc(w4, l2, c);
        let (w5, c) = adc(w5, d2, c);
        let (w6, c) = adc(w6, l3, c);
        let w7 = w7 + d3 + u64::from(c);
        reduce_wide(&[w0, w1, w2, w3, w4, w5, w6, w7])
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

#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn adc(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (s, c1) = a.overflowing_add(b);
    let (s, c2) = s.overflowing_add(u64::from(carry));
    (s, c1 | c2)
}

/// `a - b - (1 - carry)` as (difference, carry out), i.e. one step of a
/// borrow chain with the borrow kept inverted, as the flag register holds it:
/// `a - b - borrow = a + !b + (1 - borrow)`. Written as an addition because
/// the compiler turns [`adc`] chains into add-with-carry instructions but
/// does not fuse the equivalent subtraction chains.
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn sbb(a: u64, b: u64, carry: bool) -> (u64, bool) {
    adc(a, !b, carry)
}

/// `a * b` as (lo, hi).
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let t = (a as u128) * (b as u128);
    (t as u64, (t >> 64) as u64)
}

/// `w += ai * b << (64 i)` for `1 <= i <= 3`, given `w < 2^(64 (i + 4))`
/// (so `w[i + 4..]` is zero and the result fits in `w[..i + 5]`).
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn mul_row(w: &mut [u64; 8], i: usize, ai: u64, b: &[u64; 4]) {
    let (l0, h0) = mul_wide(ai, b[0]);
    let (l1, h1) = mul_wide(ai, b[1]);
    let (l2, h2) = mul_wide(ai, b[2]);
    let (l3, h3) = mul_wide(ai, b[3]);
    let (s0, c) = w[i].overflowing_add(l0);
    let (s1, c) = adc(w[i + 1], l1, c);
    let (s2, c) = adc(w[i + 2], l2, c);
    let (s3, c) = adc(w[i + 3], l3, c);
    let s4 = u64::from(c);
    let (s1, c) = s1.overflowing_add(h0);
    let (s2, c) = adc(s2, h1, c);
    let (s3, c) = adc(s3, h2, c);
    let s4 = s4 + h3 + u64::from(c);
    w[i] = s0;
    w[i + 1] = s1;
    w[i + 2] = s2;
    w[i + 3] = s3;
    w[i + 4] = s4;
}

/// Folds a 512-bit product to `r + c·2^256` with `r < 2^256`; when `c` is set
/// (probability about 2^-190) `r` is tiny.
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn fold_wide(w: &[u64; 8]) -> ([u64; 4], bool) {
    // First fold: r = lo + hi * c. hi * c < 2^290, so the carry out is < 2^35.
    let (l0, h0) = mul_wide(w[4], C);
    let (l1, h1) = mul_wide(w[5], C);
    let (l2, h2) = mul_wide(w[6], C);
    let (l3, h3) = mul_wide(w[7], C);
    let (r0, c) = w[0].overflowing_add(l0);
    let (r1, c) = adc(w[1], l1, c);
    let (r2, c) = adc(w[2], l2, c);
    let (r3, c) = adc(w[3], l3, c);
    let r4 = u64::from(c);
    let (r1, c) = r1.overflowing_add(h0);
    let (r2, c) = adc(r2, h1, c);
    let (r3, c) = adc(r3, h2, c);
    let r4 = r4 + h3 + u64::from(c);
    // Second fold: r += r4 * c (< 2^68).
    let (l, h) = mul_wide(r4, C);
    let (r0, c) = r0.overflowing_add(l);
    let (r1, c) = adc(r1, h, c);
    let (r2, c) = r2.overflowing_add(u64::from(c));
    let (r3, c) = r3.overflowing_add(u64::from(c));
    ([r0, r1, r2, r3], c)
}

/// Reduces a 512-bit little-endian product to a canonical element.
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn reduce_wide(w: &[u64; 8]) -> Fe {
    let (r, c) = fold_wide(w);
    // If the second fold carried (c), the value is 2^256 + r with r tiny, and
    // r + c is the canonical result; otherwise r < 2^256 and r + c carries
    // exactly when r >= p, giving r - p.
    let (_, c2) = add_c(&r);
    finish(r, c | c2)
}

/// Canonical element from a folded product: `r` itself, or the rare tail.
#[inline(always)]
fn finish(r: [u64; 4], rare: bool) -> Fe {
    if rare {
        fold_rare(r[0], r[1], r[2], r[3])
    } else {
        Fe(r)
    }
}

/// Rare tail of `reduce_wide` (probability about 2^-190 per product). Kept out
/// of line, with the limbs passed in registers, so the hot path is a
/// predicted-not-taken branch rather than four selects or a stack round trip.
#[cold]
#[inline(never)]
fn fold_rare(r0: u64, r1: u64, r2: u64, r3: u64) -> Fe {
    Fe(add_c(&[r0, r1, r2, r3]).0)
}

/// Out-of-line zero for the rare branches (see `fold_rare`).
#[cold]
#[inline(never)]
fn zero_cold() -> Fe {
    Fe::ZERO
}

/// `a + c` as (sum mod 2^256, carry).
#[inline(always)]
fn add_c(a: &[u64; 4]) -> ([u64; 4], bool) {
    let (t0, c0) = a[0].overflowing_add(C);
    let (t1, c1) = a[1].overflowing_add(u64::from(c0));
    let (t2, c2) = a[2].overflowing_add(u64::from(c1));
    let (t3, c3) = a[3].overflowing_add(u64::from(c2));
    ([t0, t1, t2, t3], c3)
}

/// `if cond { a } else { b }`, written limb-wise so it compiles to selects.
#[cfg(any(not(all(target_arch = "aarch64", feature = "asm")), test))]
#[inline(always)]
fn select(cond: bool, a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let m = 0u64.wrapping_sub(u64::from(cond));
    [
        b[0] ^ ((a[0] ^ b[0]) & m),
        b[1] ^ ((a[1] ^ b[1]) & m),
        b[2] ^ ((a[2] ^ b[2]) & m),
        b[3] ^ ((a[3] ^ b[3]) & m),
    ]
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

    /// The assembly multiplication and squaring agree with the portable
    /// implementation on edge cases and biased random inputs.
    #[cfg(all(target_arch = "aarch64", feature = "asm"))]
    #[test]
    fn asm_matches_generic() {
        let mut rng = Rng(0x0dd0_1234_5678_9abc);
        let mut cases = edge_cases();
        cases.extend((0..5000).map(|_| rng.fe()));
        for a in &cases {
            assert_eq!(a.square(), a.square_generic(), "{a:?}");
            assert_eq!(a.mul(a), a.square_generic(), "{a:?}");
            if !a.is_zero() {
                assert_eq!(a.neg(), a.neg_generic(), "{a:?}");
            }
            for b in cases.iter().step_by(37) {
                assert_eq!(a.mul(b), a.mul_generic(b), "{a:?} * {b:?}");
                assert_eq!(a.mul(b), b.mul(a), "{a:?} * {b:?}");
                assert_eq!(a.add(b), a.add_generic(b), "{a:?} + {b:?}");
                assert_eq!(a.sub(b), a.sub_generic(b), "{a:?} - {b:?}");
            }
        }
    }

    #[test]
    fn neg_top_limb_matches_neg() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut cases = edge_cases();
        cases.extend((0..2000).map(|_| rng.fe()));
        // Values whose low 192 bits straddle those of p: the borrow flips.
        for low in [P[0] - 1, P[0], P[0] + 1] {
            for top in [0u64, 1, 5, u64::MAX - 1] {
                cases.push(Fe([low, u64::MAX, u64::MAX, top]));
            }
        }
        for s in cases {
            if s.is_zero()
                || s.0[3] == u64::MAX && s.0[2] == u64::MAX && s.0[1] == u64::MAX && s.0[0] >= P[0]
            {
                continue;
            }
            assert_eq!(s.neg_top_limb(), s.neg().top_limb(), "{s:?}");
        }
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
        assert_eq!(Fe::BETA.add(&Fe::ONE).neg(), Fe::BETA2);
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

    fn limbs(v: &BigUint) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (slot, digit) in out.iter_mut().zip(v.to_u64_digits()) {
            *slot = digit;
        }
        out
    }

    /// The third fold of `reduce_wide` is unreachable with random inputs
    /// (probability about 2^-189): craft products with `lo + hi·c` just below
    /// `2^257` so the second fold overflows.
    #[test]
    fn reduce_wide_third_fold_is_correct() {
        let c = BigUint::from(C);
        let two256 = BigUint::one() << 256u32;
        for extra in 0u64..2000 {
            let target = (&two256 << 1u32) - BigUint::one() - BigUint::from(extra);
            let hi = (&two256 + &c - BigUint::one()) / &c + BigUint::from(extra % 7);
            let hic = &hi * &c;
            if hic > target {
                continue;
            }
            let lo = &target - &hic;
            if lo >= two256 {
                continue;
            }
            let mut w = [0u64; 8];
            w[..4].copy_from_slice(&limbs(&lo));
            w[4..].copy_from_slice(&limbs(&hi));
            let got = reduce_wide(&w);
            check_canonical(&got);
            assert_eq!(to_big(&got), (&lo + (&hi << 256u32)) % p(), "extra {extra}");
            #[cfg(all(target_arch = "aarch64", feature = "asm"))]
            {
                let (r, rare) = aarch64::fold(&w);
                assert_eq!(finish(r, rare), got, "asm fold, extra {extra}");
            }
        }
    }

    /// Differential fuzz against the big-integer reference, biased toward
    /// all-ones limbs, values next to p, high bits set and tiny values.
    #[test]
    fn heavy_fuzz_vs_bigint() {
        let pp = p();
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let biased = |rng: &mut Rng| -> Fe {
            let mut l = [rng.next(), rng.next(), rng.next(), rng.next()];
            match rng.next() % 6 {
                0 => {
                    for x in &mut l {
                        if rng.next().is_multiple_of(2) {
                            *x = u64::MAX;
                        }
                    }
                }
                1 => {
                    l = P;
                    l[0] = l[0].wrapping_sub(rng.next() % 3);
                }
                2 => {
                    l[3] |= 1 << 63;
                    l[2] = u64::MAX;
                    l[1] = u64::MAX;
                }
                3 => l = [rng.next() % 4, 0, 0, 0],
                _ => {}
            }
            let mut fe = Fe(l);
            fe.reduce_once();
            fe
        };
        for _ in 0..50_000 {
            let a = biased(&mut rng);
            let b = biased(&mut rng);
            let (ba, bb) = (to_big(&a), to_big(&b));
            assert!(ba < pp && bb < pp);
            let m = a.mul(&b);
            check_canonical(&m);
            assert_eq!(to_big(&m), (&ba * &bb) % &pp);
            assert_eq!(a.square(), a.mul(&a));
            assert_eq!(to_big(&a.add(&b)), (&ba + &bb) % &pp);
            assert_eq!(to_big(&a.sub(&b)), (&ba + &pp - &bb) % &pp);
            assert_eq!(to_big(&a.neg()), (&pp - &ba) % &pp);
            assert_eq!(Fe::from_bytes_be(&a.to_bytes_be()), a);
        }
    }
}
