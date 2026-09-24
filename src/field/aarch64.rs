//! AArch64 assembly for the hot field operations. Multiplication and
//! squaring: the schoolbook product, the two folds by `c = 2^256 mod p` and
//! the canonical check as flag chains (`adds`/`adcs`), so no carry is ever
//! materialised in a register and nothing is spilled mid-product; same
//! contract as the generic `fold_wide`: they return the folded limbs and
//! whether the rare tail (`fold_rare`, in `field.rs`) must run. Addition,
//! subtraction and negation: borrow chains as `sbcs`, which the compiler does
//! not emit from the portable code, and the canonical fix-up as one
//! conditional compare plus selects.
//!
//! Opt-in (`--features asm`) and the only `unsafe` in the crate. On an Apple
//! M3 it measured level with the portable code, which the compiler already
//! turns into the same multiply and add-with-carry chains; it is kept for
//! other AArch64 cores where that may differ. Soundness rests on the blocks being pure
//! register arithmetic: no memory access (`nomem`), no stack (`nostack`),
//! every register they touch declared as an operand, flags clobbered as
//! `asm!` assumes by default. Correctness is checked in the tests of
//! `field.rs` against the generic implementation and a big-integer reference.

use std::arch::asm;

use super::C;

/// Folds the 512-bit product in `w0..w7` into `w0..w3` (mod p, `< 2^256`) and
/// sets `f` to non-zero when the result needs the rare tail: the second fold
/// carried out of 256 bits, or `w0..w3 >= p`. Clobbers `t0..t3` and `w4`.
/// (Selecting the rare result in with `csel` instead, to make the batch loop
/// branch-free, measured 4% slower: the selects sit on the dependency chain.)
macro_rules! fold {
    () => {
        concat!(
            // r = lo + hi·c, low halves then high halves; the carry out is < 2^35.
            "mul {t0}, {w4}, {c}\n",
            "mul {t1}, {w5}, {c}\n",
            "mul {t2}, {w6}, {c}\n",
            "mul {t3}, {w7}, {c}\n",
            "adds {w0}, {w0}, {t0}\n",
            "adcs {w1}, {w1}, {t1}\n",
            "adcs {w2}, {w2}, {t2}\n",
            "adcs {w3}, {w3}, {t3}\n",
            "adc {t0}, xzr, xzr\n",
            "umulh {t1}, {w4}, {c}\n",
            "umulh {t2}, {w5}, {c}\n",
            "umulh {t3}, {w6}, {c}\n",
            "umulh {w4}, {w7}, {c}\n",
            "adds {w1}, {w1}, {t1}\n",
            "adcs {w2}, {w2}, {t2}\n",
            "adcs {w3}, {w3}, {t3}\n",
            "adc {t0}, {t0}, {w4}\n",
            // r += r4·c (< 2^68); a carry out here is the rare case.
            "mul {t1}, {t0}, {c}\n",
            "umulh {t2}, {t0}, {c}\n",
            "adds {w0}, {w0}, {t1}\n",
            "adcs {w1}, {w1}, {t2}\n",
            "adcs {w2}, {w2}, xzr\n",
            "adcs {w3}, {w3}, xzr\n",
            "cset {f}, hs\n",
            // r >= p exactly when r + c carries: the other rare case.
            "adds {t1}, {w0}, {c}\n",
            "adcs {t1}, {w1}, xzr\n",
            "adcs {t1}, {w2}, xzr\n",
            "adcs {t1}, {w3}, xzr\n",
            "cinc {f}, {f}, hs",
        )
    };
}

/// `a · b` folded: (limbs, rare).
#[inline(always)]
pub fn mul(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let (r0, r1, r2, r3, rare): (u64, u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            // Row 0: w0..w4 = a0·b.
            "mul {w0}, {a0}, {b0}",
            "umulh {t0}, {a0}, {b0}",
            "mul {w1}, {a0}, {b1}",
            "umulh {t1}, {a0}, {b1}",
            "mul {w2}, {a0}, {b2}",
            "umulh {t2}, {a0}, {b2}",
            "mul {w3}, {a0}, {b3}",
            "umulh {w4}, {a0}, {b3}",
            "adds {w1}, {w1}, {t0}",
            "adcs {w2}, {w2}, {t1}",
            "adcs {w3}, {w3}, {t2}",
            "adc {w4}, {w4}, xzr",
            // Row 1: w1..w5 += a1·b, low halves then high halves.
            "mul {t0}, {a1}, {b0}",
            "mul {t1}, {a1}, {b1}",
            "mul {t2}, {a1}, {b2}",
            "mul {t3}, {a1}, {b3}",
            "adds {w1}, {w1}, {t0}",
            "adcs {w2}, {w2}, {t1}",
            "adcs {w3}, {w3}, {t2}",
            "adcs {w4}, {w4}, {t3}",
            "adc {w5}, xzr, xzr",
            "umulh {t0}, {a1}, {b0}",
            "umulh {t1}, {a1}, {b1}",
            "umulh {t2}, {a1}, {b2}",
            "umulh {t3}, {a1}, {b3}",
            "adds {w2}, {w2}, {t0}",
            "adcs {w3}, {w3}, {t1}",
            "adcs {w4}, {w4}, {t2}",
            "adc {w5}, {w5}, {t3}",
            // Row 2: w2..w6 += a2·b.
            "mul {t0}, {a2}, {b0}",
            "mul {t1}, {a2}, {b1}",
            "mul {t2}, {a2}, {b2}",
            "mul {t3}, {a2}, {b3}",
            "adds {w2}, {w2}, {t0}",
            "adcs {w3}, {w3}, {t1}",
            "adcs {w4}, {w4}, {t2}",
            "adcs {w5}, {w5}, {t3}",
            "adc {w6}, xzr, xzr",
            "umulh {t0}, {a2}, {b0}",
            "umulh {t1}, {a2}, {b1}",
            "umulh {t2}, {a2}, {b2}",
            "umulh {t3}, {a2}, {b3}",
            "adds {w3}, {w3}, {t0}",
            "adcs {w4}, {w4}, {t1}",
            "adcs {w5}, {w5}, {t2}",
            "adc {w6}, {w6}, {t3}",
            // Row 3: w3..w7 += a3·b.
            "mul {t0}, {a3}, {b0}",
            "mul {t1}, {a3}, {b1}",
            "mul {t2}, {a3}, {b2}",
            "mul {t3}, {a3}, {b3}",
            "adds {w3}, {w3}, {t0}",
            "adcs {w4}, {w4}, {t1}",
            "adcs {w5}, {w5}, {t2}",
            "adcs {w6}, {w6}, {t3}",
            "adc {w7}, xzr, xzr",
            "umulh {t0}, {a3}, {b0}",
            "umulh {t1}, {a3}, {b1}",
            "umulh {t2}, {a3}, {b2}",
            "umulh {t3}, {a3}, {b3}",
            "adds {w4}, {w4}, {t0}",
            "adcs {w5}, {w5}, {t1}",
            "adcs {w6}, {w6}, {t2}",
            "adc {w7}, {w7}, {t3}",
            fold!(),
            a0 = in(reg) a[0],
            a1 = in(reg) a[1],
            a2 = in(reg) a[2],
            a3 = in(reg) a[3],
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            c = in(reg) C,
            w0 = out(reg) r0,
            w1 = out(reg) r1,
            w2 = out(reg) r2,
            w3 = out(reg) r3,
            w4 = out(reg) _,
            w5 = out(reg) _,
            w6 = out(reg) _,
            w7 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            f = out(reg) rare,
            options(pure, nomem, nostack),
        );
    }
    ([r0, r1, r2, r3], rare != 0)
}

/// `a²` folded: (limbs, rare). Off-diagonal products once, doubled with an
/// add chain, then the diagonal.
#[inline(always)]
pub fn square(a: &[u64; 4]) -> ([u64; 4], bool) {
    let (r0, r1, r2, r3, rare): (u64, u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            // Off-diagonal products a_i·a_j, i < j, each once, into w1..w6.
            "mul {w1}, {a0}, {a1}",
            "umulh {t0}, {a0}, {a1}",
            "mul {w2}, {a0}, {a2}",
            "umulh {t1}, {a0}, {a2}",
            "mul {w3}, {a0}, {a3}",
            "umulh {w4}, {a0}, {a3}",
            "adds {w2}, {w2}, {t0}",
            "adcs {w3}, {w3}, {t1}",
            "adc {w4}, {w4}, xzr",
            "mul {t0}, {a1}, {a2}",
            "umulh {t1}, {a1}, {a2}",
            "mul {t2}, {a1}, {a3}",
            "umulh {t3}, {a1}, {a3}",
            "adds {w3}, {w3}, {t0}",
            "adcs {w4}, {w4}, {t2}",
            "adc {w5}, xzr, xzr",
            "adds {w4}, {w4}, {t1}",
            "adc {w5}, {w5}, {t3}",
            "mul {t0}, {a2}, {a3}",
            "umulh {w6}, {a2}, {a3}",
            "adds {w5}, {w5}, {t0}",
            "adc {w6}, {w6}, xzr",
            // Double (the sum is < 2^448, so w7 is the shifted-out bit).
            "adds {w1}, {w1}, {w1}",
            "adcs {w2}, {w2}, {w2}",
            "adcs {w3}, {w3}, {w3}",
            "adcs {w4}, {w4}, {w4}",
            "adcs {w5}, {w5}, {w5}",
            "adcs {w6}, {w6}, {w6}",
            "adc {w7}, xzr, xzr",
            // Add the squares on the diagonal.
            "mul {w0}, {a0}, {a0}",
            "umulh {t0}, {a0}, {a0}",
            "mul {t1}, {a1}, {a1}",
            "umulh {t2}, {a1}, {a1}",
            "adds {w1}, {w1}, {t0}",
            "adcs {w2}, {w2}, {t1}",
            "adcs {w3}, {w3}, {t2}",
            "mul {t0}, {a2}, {a2}",
            "umulh {t1}, {a2}, {a2}",
            "mul {t2}, {a3}, {a3}",
            "umulh {t3}, {a3}, {a3}",
            "adcs {w4}, {w4}, {t0}",
            "adcs {w5}, {w5}, {t1}",
            "adcs {w6}, {w6}, {t2}",
            "adc {w7}, {w7}, {t3}",
            fold!(),
            a0 = in(reg) a[0],
            a1 = in(reg) a[1],
            a2 = in(reg) a[2],
            a3 = in(reg) a[3],
            c = in(reg) C,
            w0 = out(reg) r0,
            w1 = out(reg) r1,
            w2 = out(reg) r2,
            w3 = out(reg) r3,
            w4 = out(reg) _,
            w5 = out(reg) _,
            w6 = out(reg) _,
            w7 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            f = out(reg) rare,
            options(pure, nomem, nostack),
        );
    }
    ([r0, r1, r2, r3], rare != 0)
}

/// The fold alone, on a 512-bit little-endian product: (limbs, rare). Only
/// used by the tests, to check the reduction on crafted products the
/// multiplications never produce.
#[cfg(test)]
pub fn fold(w: &[u64; 8]) -> ([u64; 4], bool) {
    let (r0, r1, r2, r3, rare): (u64, u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            fold!(),
            c = in(reg) C,
            w0 = inout(reg) w[0] => r0,
            w1 = inout(reg) w[1] => r1,
            w2 = inout(reg) w[2] => r2,
            w3 = inout(reg) w[3] => r3,
            w4 = inout(reg) w[4] => _,
            w5 = in(reg) w[5],
            w6 = in(reg) w[6],
            w7 = in(reg) w[7],
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            f = out(reg) rare,
            options(pure, nomem, nostack),
        );
    }
    ([r0, r1, r2, r3], rare != 0)
}

/// `a + b mod p` for canonical inputs: the sum, minus `p` (i.e. plus `c` mod
/// 2^256) when it carried out of 256 bits or is still `>= p`.
#[inline(always)]
pub fn add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (r0, r1, r2, r3): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            "adds {r0}, {a0}, {b0}",
            "adcs {r1}, {a1}, {b1}",
            "adcs {r2}, {a2}, {b2}",
            "adcs {r3}, {a3}, {b3}",
            "cset {f}, hs",
            // t = r + c; r >= p exactly when this carries.
            "adds {t0}, {r0}, {c}",
            "adcs {t1}, {r1}, xzr",
            "adcs {t2}, {r2}, xzr",
            "adcs {t3}, {r3}, xzr",
            // Flags := (carried here) ? "ne" : (f != 0).
            "ccmp {f}, #0, #0, lo",
            "csel {r0}, {t0}, {r0}, ne",
            "csel {r1}, {t1}, {r1}, ne",
            "csel {r2}, {t2}, {r2}, ne",
            "csel {r3}, {t3}, {r3}, ne",
            a0 = in(reg) a[0],
            a1 = in(reg) a[1],
            a2 = in(reg) a[2],
            a3 = in(reg) a[3],
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            c = in(reg) C,
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            f = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// `a - b mod p` for canonical inputs: the difference, plus `p` (i.e. minus
/// `c` from the wrapped value) when it borrowed.
#[inline(always)]
pub fn sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (r0, r1, r2, r3): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            "subs {r0}, {a0}, {b0}",
            "sbcs {r1}, {a1}, {b1}",
            "sbcs {r2}, {a2}, {b2}",
            "sbcs {r3}, {a3}, {b3}",
            "csel {t}, {c}, xzr, lo",
            "subs {r0}, {r0}, {t}",
            "sbcs {r1}, {r1}, xzr",
            "sbcs {r2}, {r2}, xzr",
            "sbc {r3}, {r3}, xzr",
            a0 = in(reg) a[0],
            a1 = in(reg) a[1],
            a2 = in(reg) a[2],
            a3 = in(reg) a[3],
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            c = in(reg) C,
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            t = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// `p - s` for canonical non-zero `s`: `2^256 - s` (which is `> c`), minus
/// `c`.
#[inline(always)]
pub fn neg(s: &[u64; 4]) -> [u64; 4] {
    let (r0, r1, r2, r3): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic; all registers written are outputs.
    unsafe {
        asm!(
            "negs {r0}, {s0}",
            "ngcs {r1}, {s1}",
            "ngcs {r2}, {s2}",
            "ngc {r3}, {s3}",
            "subs {r0}, {r0}, {c}",
            "sbcs {r1}, {r1}, xzr",
            "sbcs {r2}, {r2}, xzr",
            "sbc {r3}, {r3}, xzr",
            s0 = in(reg) s[0],
            s1 = in(reg) s[1],
            s2 = in(reg) s[2],
            s3 = in(reg) s[3],
            c = in(reg) C,
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}
