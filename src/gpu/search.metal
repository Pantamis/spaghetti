// secp256k1 x-only batched walk on the GPU: the Metal port of `Walk::batch`
// and `Searcher` (src/search.rs). One thread owns one walk; per batch it
// computes the x coordinates of `C ± j·G` for `j = 1..=H` with a single field
// inversion (Montgomery's trick, prefix products kept in a per-thread slice
// of `scratch`), tests `x`, `β·x` and `β²·x` of every point against the
// pattern keys (top 64 bits of x), appends hits to `hits`, and finally jumps
// `C += (2H+1)·G`. The host owns the scalars: a thread's centre after `b`
// batches is `base + (k0[t] + b·(2H+1))·G`, so a hit record needs only the
// thread, the batch index within the dispatch, the signed offset and the
// endomorphism power.
//
// Field arithmetic: canonical (`< p`) 8×32-bit little-endian limbs, since the
// GPU ALUs are 32-bit. The product is one Karatsuba level over 128-bit
// halves of column-wise (Comba) 32×32→64 limb products, reduced with two
// folds by `c = 2^256 mod p = 2^32 + 977` and one conditional subtraction of
// `p`, exactly as `reduce_wide` in src/field.rs. Measured on an Apple M3 Pro
// against the alternatives (see the `bench` kernel): a 32×32→64 product
// costs the same as `mulhi` alone, 64-bit adds are emulated and cost about a
// multiply, and Karatsuba is 11% faster than the plain 64-product schoolbook.
// The tests compare every operation against the CPU implementation.

#include <metal_stdlib>
using namespace metal;

/// `scratch` entry `(j, t)`: `uint4` offset of the walk's j-th prefix product.
#define SCRATCH(j) (2 * ((j) * n + t))

struct Fe {
    uint v[8];
};

/// Low limb of `p`; limbs 1..=7 are all ones.
constant uint P0 = 0xFFFFFC2Fu;
/// `c = 2^256 mod p = 2^32 + 977`: 977 at limb 0, 1 at limb 1.
constant uint C_LO = 977u;

/// β (cube root of unity in the field): λ·(x, y) = (β·x, y).
constant Fe BETA = {{0x719501eeu, 0xc1396c28u, 0x12f58995u, 0x9cf04975u,
                     0xac3434e9u, 0x6e64479eu, 0x657c0710u, 0x7ae96a2bu}};

struct Params {
    uint n;          // walks (threads)
    uint h;          // H: points on each side of the centre per batch
    uint batches;    // batches per thread this dispatch (upper bound)
    uint nkeys;      // pattern keys
    uint hit_cap;    // capacity of `hits`
    uint first_only; // 1: a thread stops at its first hit (random mode)
};

/// Top 64 bits of x as (limb 7, limb 6) masks and values.
struct Key {
    uint mask_hi, mask_lo, value_hi, value_lo;
};

struct Hit {
    uint walk;    // thread index
    uint batch;   // index of the batch within the dispatch
    int offset;   // −H..=H
    uint endo;    // power of λ applied: 0, 1, 2
    uint x[8];    // the matching x coordinate (λ^endo · x), for the host check
};

// ---- field arithmetic -------------------------------------------------------

inline Fe fe_load(const device uint4* p) {
    uint4 a = p[0], b = p[1];
    Fe r = {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
    return r;
}

inline Fe fe_load_c(const constant uint4* p) {
    uint4 a = p[0], b = p[1];
    Fe r = {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
    return r;
}

inline void fe_store(device uint4* p, Fe f) {
    p[0] = uint4(f.v[0], f.v[1], f.v[2], f.v[3]);
    p[1] = uint4(f.v[4], f.v[5], f.v[6], f.v[7]);
}

inline bool fe_is_zero(Fe a) {
    uint acc = 0;
    for (uint i = 0; i < 8; i++) {
        acc |= a.v[i];
    }
    return acc == 0;
}

/// `a + b mod p`.
inline Fe fe_add(Fe a, Fe b) {
    Fe r, s;
    ulong acc = 0;
    for (uint i = 0; i < 8; i++) {
        acc += (ulong)a.v[i] + (ulong)b.v[i];
        r.v[i] = (uint)acc;
        acc >>= 32;
    }
    uint carry = (uint)acc;
    // r >= p exactly when r + c carries out of 256 bits, and then r − p is
    // r + c mod 2^256; when the sum itself carried, r + c is the answer too.
    acc = (ulong)r.v[0] + C_LO;
    s.v[0] = (uint)acc;
    acc >>= 32;
    acc += (ulong)r.v[1] + 1u;
    s.v[1] = (uint)acc;
    acc >>= 32;
    for (uint i = 2; i < 8; i++) {
        acc += r.v[i];
        s.v[i] = (uint)acc;
        acc >>= 32;
    }
    bool fix = (carry | (uint)acc) != 0;
    for (uint i = 0; i < 8; i++) {
        r.v[i] = fix ? s.v[i] : r.v[i];
    }
    return r;
}

/// `a − b mod p`.
inline Fe fe_sub(Fe a, Fe b) {
    Fe r;
    long acc = 0;
    for (uint i = 0; i < 8; i++) {
        acc += (long)a.v[i] - (long)b.v[i];
        r.v[i] = (uint)acc;
        acc >>= 32; // arithmetic: −1 carries the borrow
    }
    // Wrapped below zero: add p back, i.e. subtract c from the wrapped value
    // (which is > c, so this cannot underflow).
    bool borrow = acc != 0;
    acc = (long)r.v[0] - (borrow ? (long)C_LO : 0l);
    r.v[0] = (uint)acc;
    acc >>= 32;
    acc += (long)r.v[1] - (borrow ? 1l : 0l);
    r.v[1] = (uint)acc;
    acc >>= 32;
    for (uint i = 2; i < 8; i++) {
        acc += (long)r.v[i];
        r.v[i] = (uint)acc;
        acc >>= 32;
    }
    return r;
}

inline Fe fe_neg(Fe a) {
    Fe zero = {{0, 0, 0, 0, 0, 0, 0, 0}};
    return fe_sub(zero, a);
}

/// Reduces the 512-bit product `w` to a canonical element: two folds by `c`
/// and one conditional subtraction of `p` (see `reduce_wide` in field.rs).
/// 32-bit adds with the carries counted: 64-bit adds are emulated on Apple
/// GPUs and measured about as costly as a multiply.
inline Fe fe_reduce(thread const uint* w) {
    Fe r, s;
    // First fold: r = lo + hi·977 + (hi << 32). Limb i sums w[i],
    // low(hi[i]·977), high(hi[i−1]·977), hi[i−1] (the shift) and the carry
    // count of limb i−1: five words, so the carry count stays below 5 and
    // the fold's total carry t below 2^33 (as `fold_wide`).
    uint c = 0;
    uint hprev = 0;
    for (uint i = 0; i < 8; i++) {
        ulong p = (ulong)w[8 + i] * (ulong)C_LO;
        uint l = (uint)p;
        uint acc = w[i];
        uint cn = 0;
        acc += l;
        cn += (acc < l) ? 1u : 0u;
        acc += hprev;
        cn += (acc < hprev) ? 1u : 0u;
        uint sh = (i == 0) ? 0u : w[7 + i];
        acc += sh;
        cn += (acc < sh) ? 1u : 0u;
        acc += c;
        cn += (acc < c) ? 1u : 0u;
        r.v[i] = acc;
        c = cn;
        hprev = (uint)(p >> 32);
    }
    // t = c + hprev + w[15] (the top of the shift): t < 2^33.
    uint t_lo = c + hprev;
    uint t_hi = (t_lo < hprev) ? 1u : 0u;
    t_lo += w[15];
    t_hi += (t_lo < w[15]) ? 1u : 0u;
    // Second fold: r += t·977 + (t << 32): limb 0 += low(t_lo·977),
    // limb 1 += high(t_lo·977) + t_hi·977 + t_lo, limb 2 += t_hi, then the
    // carries; the whole thing carries out of 256 bits at most once (c2).
    ulong p = (ulong)t_lo * (ulong)C_LO;
    uint l = (uint)p;
    uint h = (uint)(p >> 32) + t_hi * C_LO;
    r.v[0] += l;
    uint cy = (r.v[0] < l) ? 1u : 0u;
    uint c1 = 0;
    r.v[1] += h;
    c1 += (r.v[1] < h) ? 1u : 0u;
    r.v[1] += t_lo;
    c1 += (r.v[1] < t_lo) ? 1u : 0u;
    r.v[1] += cy;
    c1 += (r.v[1] < cy) ? 1u : 0u;
    uint add2 = c1 + t_hi;
    r.v[2] += add2;
    cy = (r.v[2] < add2) ? 1u : 0u;
    for (uint i = 3; i < 8; i++) {
        r.v[i] += cy;
        cy = (r.v[i] < cy) ? 1u : 0u;
    }
    uint c2 = cy;
    // Canonical: s = r + c; r + c carries exactly when r >= p (then r − p is
    // s mod 2^256); when the second fold carried, r is tiny and s is the
    // answer too.
    s.v[0] = r.v[0] + C_LO;
    cy = (s.v[0] < C_LO) ? 1u : 0u;
    uint one = 1u + cy;
    s.v[1] = r.v[1] + one;
    cy = (s.v[1] < one) ? 1u : 0u;
    for (uint i = 2; i < 8; i++) {
        s.v[i] = r.v[i] + cy;
        cy = (s.v[i] < cy) ? 1u : 0u;
    }
    bool fix = (c2 | cy) != 0;
    for (uint i = 0; i < 8; i++) {
        r.v[i] = fix ? s.v[i] : r.v[i];
    }
    return r;
}

// Column-wise (Comba) schoolbook product. Each limb pair is one 32×32→64
// product (measured to cost the same as `mulhi` alone, so the low half is
// free); the low and high halves of a column's products are summed in two
// 64-bit accumulators (at most 8 terms each, no overflow), then the column is
// closed. Generated, fully unrolled so every limb stays in a register.
#define MULADD(i, j)                                  \
    {                                                 \
        ulong p = (ulong)a.v[i] * (ulong)b.v[j];      \
        lo += (uint)p;                                \
        hi += (uint)(p >> 32);                        \
    }
#define COLUMN(k)                                     \
    acc += lo;                                        \
    w[k] = (uint)acc;                                 \
    acc = (acc >> 32) + hi;                           \
    lo = 0;                                           \
    hi = 0;

inline Fe fe_mul_plain(Fe a, Fe b) {
    uint w[16];
    ulong acc = 0, lo = 0, hi = 0;
    MULADD(0, 0);
    COLUMN(0);
    MULADD(0, 1);
    MULADD(1, 0);
    COLUMN(1);
    MULADD(0, 2);
    MULADD(1, 1);
    MULADD(2, 0);
    COLUMN(2);
    MULADD(0, 3);
    MULADD(1, 2);
    MULADD(2, 1);
    MULADD(3, 0);
    COLUMN(3);
    MULADD(0, 4);
    MULADD(1, 3);
    MULADD(2, 2);
    MULADD(3, 1);
    MULADD(4, 0);
    COLUMN(4);
    MULADD(0, 5);
    MULADD(1, 4);
    MULADD(2, 3);
    MULADD(3, 2);
    MULADD(4, 1);
    MULADD(5, 0);
    COLUMN(5);
    MULADD(0, 6);
    MULADD(1, 5);
    MULADD(2, 4);
    MULADD(3, 3);
    MULADD(4, 2);
    MULADD(5, 1);
    MULADD(6, 0);
    COLUMN(6);
    MULADD(0, 7);
    MULADD(1, 6);
    MULADD(2, 5);
    MULADD(3, 4);
    MULADD(4, 3);
    MULADD(5, 2);
    MULADD(6, 1);
    MULADD(7, 0);
    COLUMN(7);
    MULADD(1, 7);
    MULADD(2, 6);
    MULADD(3, 5);
    MULADD(4, 4);
    MULADD(5, 3);
    MULADD(6, 2);
    MULADD(7, 1);
    COLUMN(8);
    MULADD(2, 7);
    MULADD(3, 6);
    MULADD(4, 5);
    MULADD(5, 4);
    MULADD(6, 3);
    MULADD(7, 2);
    COLUMN(9);
    MULADD(3, 7);
    MULADD(4, 6);
    MULADD(5, 5);
    MULADD(6, 4);
    MULADD(7, 3);
    COLUMN(10);
    MULADD(4, 7);
    MULADD(5, 6);
    MULADD(6, 5);
    MULADD(7, 4);
    COLUMN(11);
    MULADD(5, 7);
    MULADD(6, 6);
    MULADD(7, 5);
    COLUMN(12);
    MULADD(6, 7);
    MULADD(7, 6);
    COLUMN(13);
    MULADD(7, 7);
    COLUMN(14);
    w[15] = (uint)acc;
    return fe_reduce(w);
}

#undef MULADD
#undef COLUMN

/// The compiler shares the 28 symmetric products of `a·a` (it canonicalises
/// commutative operands), which measured faster than a hand-written
/// squaring with explicit doubling.
inline Fe fe_sqr(Fe a) {
    return fe_mul_plain(a, a);
}

// ---- Karatsuba product ------------------------------------------------------
#define K_MULADD(x, i, y, j)                          \
    {                                                 \
        ulong p = (ulong)x[i] * (ulong)y[j];          \
        lo += (uint)p;                                \
        hi += (uint)(p >> 32);                        \
    }
#define K_COLUMN(w, k)                                \
    acc += lo;                                        \
    w[k] = (uint)acc;                                 \
    acc = (acc >> 32) + hi;                           \
    lo = 0;                                           \
    hi = 0;

/// `a·b` with one Karatsuba level over the 128-bit halves: 48 limb products
/// instead of 64, for about 130 extra 32-bit additions; measured 11% faster
/// than the plain product. Squaring keeps the plain product, where the
/// compiler shares the symmetric products.
inline Fe fe_mul(Fe a, Fe b) {
    thread uint* a0 = a.v;
    thread uint* a1 = a.v + 4;
    thread uint* b0 = b.v;
    thread uint* b1 = b.v + 4;
    uint z0[8], z2[8], m[8];
    ulong acc, lo, hi;
    acc = 0; lo = 0; hi = 0;
    K_MULADD(a0, 0, b0, 0);
    K_COLUMN(z0, 0);
    K_MULADD(a0, 0, b0, 1);
    K_MULADD(a0, 1, b0, 0);
    K_COLUMN(z0, 1);
    K_MULADD(a0, 0, b0, 2);
    K_MULADD(a0, 1, b0, 1);
    K_MULADD(a0, 2, b0, 0);
    K_COLUMN(z0, 2);
    K_MULADD(a0, 0, b0, 3);
    K_MULADD(a0, 1, b0, 2);
    K_MULADD(a0, 2, b0, 1);
    K_MULADD(a0, 3, b0, 0);
    K_COLUMN(z0, 3);
    K_MULADD(a0, 1, b0, 3);
    K_MULADD(a0, 2, b0, 2);
    K_MULADD(a0, 3, b0, 1);
    K_COLUMN(z0, 4);
    K_MULADD(a0, 2, b0, 3);
    K_MULADD(a0, 3, b0, 2);
    K_COLUMN(z0, 5);
    K_MULADD(a0, 3, b0, 3);
    K_COLUMN(z0, 6);
    z0[7] = (uint)acc;
    acc = 0; lo = 0; hi = 0;
    K_MULADD(a1, 0, b1, 0);
    K_COLUMN(z2, 0);
    K_MULADD(a1, 0, b1, 1);
    K_MULADD(a1, 1, b1, 0);
    K_COLUMN(z2, 1);
    K_MULADD(a1, 0, b1, 2);
    K_MULADD(a1, 1, b1, 1);
    K_MULADD(a1, 2, b1, 0);
    K_COLUMN(z2, 2);
    K_MULADD(a1, 0, b1, 3);
    K_MULADD(a1, 1, b1, 2);
    K_MULADD(a1, 2, b1, 1);
    K_MULADD(a1, 3, b1, 0);
    K_COLUMN(z2, 3);
    K_MULADD(a1, 1, b1, 3);
    K_MULADD(a1, 2, b1, 2);
    K_MULADD(a1, 3, b1, 1);
    K_COLUMN(z2, 4);
    K_MULADD(a1, 2, b1, 3);
    K_MULADD(a1, 3, b1, 2);
    K_COLUMN(z2, 5);
    K_MULADD(a1, 3, b1, 3);
    K_COLUMN(z2, 6);
    z2[7] = (uint)acc;
    // sa = a0 + a1, sb = b0 + b1 (4 limbs + carry bit each).
    uint sa[4], sb[4];
    uint ca = 0, cb = 0;
    for (uint i = 0; i < 4; i++) {
        uint x = a0[i] + ca;
        uint c = (x < ca) ? 1u : 0u;
        x += a1[i];
        c += (x < a1[i]) ? 1u : 0u;
        sa[i] = x;
        ca = c;
        uint y = b0[i] + cb;
        uint d = (y < cb) ? 1u : 0u;
        y += b1[i];
        d += (y < b1[i]) ? 1u : 0u;
        sb[i] = y;
        cb = d;
    }
    acc = 0; lo = 0; hi = 0;
    K_MULADD(sa, 0, sb, 0);
    K_COLUMN(m, 0);
    K_MULADD(sa, 0, sb, 1);
    K_MULADD(sa, 1, sb, 0);
    K_COLUMN(m, 1);
    K_MULADD(sa, 0, sb, 2);
    K_MULADD(sa, 1, sb, 1);
    K_MULADD(sa, 2, sb, 0);
    K_COLUMN(m, 2);
    K_MULADD(sa, 0, sb, 3);
    K_MULADD(sa, 1, sb, 2);
    K_MULADD(sa, 2, sb, 1);
    K_MULADD(sa, 3, sb, 0);
    K_COLUMN(m, 3);
    K_MULADD(sa, 1, sb, 3);
    K_MULADD(sa, 2, sb, 2);
    K_MULADD(sa, 3, sb, 1);
    K_COLUMN(m, 4);
    K_MULADD(sa, 2, sb, 3);
    K_MULADD(sa, 3, sb, 2);
    K_COLUMN(m, 5);
    K_MULADD(sa, 3, sb, 3);
    K_COLUMN(m, 6);
    m[7] = (uint)acc;
    // m += ca·(sb << 128) + cb·(sa << 128) + (ca & cb) << 256; m < 2^258.
    uint m8 = ca & cb;
    uint cy = 0;
    for (uint i = 0; i < 4; i++) {
        uint x = m[4 + i] + cy;
        cy = (x < cy) ? 1u : 0u;
        uint u = ca ? sb[i] : 0u;
        x += u;
        cy += (x < u) ? 1u : 0u;
        uint v = cb ? sa[i] : 0u;
        x += v;
        cy += (x < v) ? 1u : 0u;
        m[4 + i] = x;
    }
    m8 += cy;
    // z1 = m − z0 − z2 (>= 0): 9 limbs, borrow-chain subtractions.
    long bw = 0;
    uint z1[9];
    for (uint i = 0; i < 8; i++) {
        bw += (long)m[i] - (long)z0[i] - (long)z2[i];
        z1[i] = (uint)bw;
        bw >>= 32;
    }
    z1[8] = (uint)((long)m8 + bw);
    // w = z0 + z1 << 128 + z2 << 256.
    uint w[16];
    for (uint i = 0; i < 4; i++) {
        w[i] = z0[i];
    }
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        carry += (ulong)z0[4 + i] + (ulong)z1[i];
        w[4 + i] = (uint)carry;
        carry >>= 32;
    }
    for (uint i = 0; i < 4; i++) {
        carry += (ulong)z2[i] + (ulong)z1[4 + i];
        w[8 + i] = (uint)carry;
        carry >>= 32;
    }
    carry += (ulong)z2[4] + (ulong)z1[8];
    w[12] = (uint)carry;
    carry >>= 32;
    for (uint i = 5; i < 8; i++) {
        carry += (ulong)z2[i];
        w[8 + i] = (uint)carry;
        carry >>= 32;
    }
    return fe_reduce(w);
}
#undef K_MULADD
#undef K_COLUMN


inline Fe fe_sqn(Fe a, uint n) {
    for (uint i = 0; i < n; i++) {
        a = fe_sqr(a);
    }
    return a;
}

/// `a^(p−2)` with libsecp256k1's addition chain (`Fe::invert` in field.rs).
inline Fe fe_inv(Fe x) {
    Fe x2 = fe_mul(fe_sqr(x), x);
    Fe x3 = fe_mul(fe_sqr(x2), x);
    Fe x6 = fe_mul(fe_sqn(x3, 3), x3);
    Fe x9 = fe_mul(fe_sqn(x6, 3), x3);
    Fe x11 = fe_mul(fe_sqn(x9, 2), x2);
    Fe x22 = fe_mul(fe_sqn(x11, 11), x11);
    Fe x44 = fe_mul(fe_sqn(x22, 22), x22);
    Fe x88 = fe_mul(fe_sqn(x44, 44), x44);
    Fe x176 = fe_mul(fe_sqn(x88, 88), x88);
    Fe x220 = fe_mul(fe_sqn(x176, 44), x44);
    Fe x223 = fe_mul(fe_sqn(x220, 3), x3);
    Fe t = fe_mul(fe_sqn(x223, 23), x22);
    t = fe_mul(fe_sqn(t, 5), x);
    t = fe_mul(fe_sqn(t, 3), x2);
    return fe_mul(fe_sqn(t, 2), x);
}

// ---- pattern test -----------------------------------------------------------

/// Does the top 64 bits `(hi, lo)` of an x coordinate match any key?
inline bool key_hit(uint hi, uint lo, constant Key* keys, uint nkeys) {
    bool hit = false;
    for (uint k = 0; k < nkeys; k++) {
        Key key = keys[k];
        hit |= ((hi & key.mask_hi) == key.value_hi) & ((lo & key.mask_lo) == key.value_lo);
    }
    return hit;
}

/// Top 64 bits of `p − s` for non-zero `s` (the top of `β²·x = −(x + β·x)`):
/// `p − s` borrows into the top limbs exactly when the low 192 bits of `s`
/// exceed those of `p`, i.e. limbs 1..=5 all ones and limb 0 above `P0`.
inline uint2 neg_top(Fe s) {
    uint ones = s.v[1] & s.v[2] & s.v[3] & s.v[4] & s.v[5];
    uint borrow = (ones == 0xFFFFFFFFu && s.v[0] > P0) ? 1u : 0u;
    uint lo = ~s.v[6] - borrow;
    uint hi = ~s.v[7] - ((borrow != 0 && s.v[6] == 0xFFFFFFFFu) ? 1u : 0u);
    return uint2(hi, lo);
}

inline void record(device Hit* hits, device atomic_uint* counters, uint cap, uint t, uint b, int offset,
                   uint endo, Fe x) {
    uint idx = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
    if (idx < cap) {
        Hit h;
        h.walk = t;
        h.batch = b;
        h.offset = offset;
        h.endo = endo;
        for (uint i = 0; i < 8; i++) {
            h.x[i] = x.v[i];
        }
        hits[idx] = h;
    }
}

/// Tests `x`, `β·x`, `β²·x` of one point; records the hits. Returns whether
/// anything hit.
inline bool test_point(Fe x, int offset, uint t, uint b, constant Key* keys, uint nkeys, device Hit* hits,
                       device atomic_uint* counters, uint cap) {
    Fe bx = fe_mul(BETA, x);
    Fe s = fe_add(bx, x);
    uint2 b2 = neg_top(s);
    bool h0 = key_hit(x.v[7], x.v[6], keys, nkeys);
    bool h1 = key_hit(bx.v[7], bx.v[6], keys, nkeys);
    bool h2 = key_hit(b2.x, b2.y, keys, nkeys);
    if (h0 | h1 | h2) {
        if (h0) record(hits, counters, cap, t, b, offset, 0, x);
        if (h1) record(hits, counters, cap, t, b, offset, 1, bx);
        if (h2) record(hits, counters, cap, t, b, offset, 2, fe_neg(s));
        return true;
    }
    return false;
}

// ---- the walk ---------------------------------------------------------------

/// `table`: `(H+1)` entries of `x[8], y[8]` (`j·G` for `j = 1..=H`, then the
/// jump `(2H+1)·G`). `scratch`: `(H+1)·n` elements, entry `(j, t)` at
/// `j·n + t` so a SIMD group's accesses are contiguous. `centres`: `n` entries
/// of `x[8], y[8]`. `counters`: `[hits appended, degenerate threads]`.
/// `remaining[t]`: batches this thread may still run; `flags[t]`: set to 1
/// when the thread hit a zero difference (the host recomputes its centre).
kernel void search(device uint4* centres [[buffer(0)]],
                   constant uint4* table [[buffer(1)]],
                   device uint4* scratch [[buffer(2)]],
                   constant Params& prm [[buffer(3)]],
                   constant Key* keys [[buffer(4)]],
                   device Hit* hits [[buffer(5)]],
                   device atomic_uint* counters [[buffer(6)]],
                   device uint* remaining [[buffer(7)]],
                   device uint* flags [[buffer(8)]],
                   uint t [[thread_position_in_grid]]) {
    const uint n = prm.n;
    const uint h = prm.h;
    if (t >= n) {
        return;
    }
    uint rem = remaining[t];
    if (rem == 0) {
        return;
    }
    Fe cx = fe_load(centres + 4 * t);
    Fe cy = fe_load(centres + 4 * t + 2);
    const Fe jump_x = fe_load_c(table + 4 * h);
    const uint todo = min(rem, prm.batches);
    for (uint b = 0; b < todo; b++) {
        // Forward pass over the differences, the jump first so that the
        // backward pass reaches its inverse last (nothing about the jump has
        // to stay live across the pair loop): scratch[0] = jump.x − cx,
        // scratch[j] = scratch[j−1] · (T[j−1].x − cx) for j = 1..=H.
        Fe acc = fe_sub(jump_x, cx);
        fe_store(scratch + SCRATCH(0), acc);
        for (uint j = 1; j <= h; j++) {
            acc = fe_mul(acc, fe_sub(fe_load_c(table + 4 * (j - 1)), cx));
            fe_store(scratch + SCRATCH(j), acc);
        }
        if (fe_is_zero(acc)) {
            // C == ±j·G for some table j: the batch formulas do not apply.
            flags[t] = 1;
            atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
            break;
        }
        Fe inv = fe_inv(acc);
        bool hit = test_point(cx, 0, t, b, keys, prm.nkeys, hits, counters, prm.hit_cap);
        Fe neg_cy = fe_neg(cy);
        // Backward pass: element j (table point j−1) gets inv · scratch[j−1].
        for (uint j = h; j > 0; j--) {
            Fe tx = fe_load_c(table + 4 * (j - 1));
            Fe ty = fe_load_c(table + 4 * (j - 1) + 2);
            Fe inv_here = fe_mul(inv, fe_load(scratch + SCRATCH(j - 1)));
            inv = fe_mul(inv, fe_sub(tx, cx));
            // x(C ± T) = λ² − cx − tx with λ = (±ty − cy) / (tx − cx).
            Fe sum = fe_add(cx, tx);
            Fe lambda_plus = fe_mul(fe_sub(ty, cy), inv_here);
            Fe lambda_minus = fe_mul(fe_sub(ty, neg_cy), inv_here);
            Fe x_plus = fe_sub(fe_sqr(lambda_plus), sum);
            Fe x_minus = fe_sub(fe_sqr(lambda_minus), sum);
            int offset = (int)j;
            hit |= test_point(x_plus, offset, t, b, keys, prm.nkeys, hits, counters, prm.hit_cap);
            hit |= test_point(x_minus, -offset, t, b, keys, prm.nkeys, hits, counters, prm.hit_cap);
        }
        // Jump: C += (2H+1)·G, with inv = 1 / (jump.x − cx).
        Fe jump_y = fe_load_c(table + 4 * h + 2);
        Fe lambda = fe_mul(fe_sub(jump_y, cy), inv);
        Fe nx = fe_sub(fe_sub(fe_sqr(lambda), cx), jump_x);
        Fe ny = fe_sub(fe_mul(lambda, fe_sub(cx, nx)), cy);
        cx = nx;
        cy = ny;
        rem--;
        if (hit && prm.first_only != 0) {
            // Random mode: report this batch's hits only; the host reseeds.
            break;
        }
    }
    fe_store(centres + 4 * t, cx);
    fe_store(centres + 4 * t + 2, cy);
    remaining[t] = rem;
}


// ---- self-test kernel -------------------------------------------------------

/// `out[i] = [a·b, a², a+b, a−b, −a, a⁻¹]` for `in[i] = [a, b]`; the tests
/// compare it with the CPU field.
kernel void field_test(const device uint4* in [[buffer(0)]],
                       device uint4* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    Fe a = fe_load(in + 4 * i);
    Fe b = fe_load(in + 4 * i + 2);
    device uint4* o = out + 12 * i;
    fe_store(o, fe_mul(a, b));
    fe_store(o + 2, fe_sqr(a));
    fe_store(o + 4, fe_add(a, b));
    fe_store(o + 6, fe_sub(a, b));
    fe_store(o + 8, fe_neg(a));
    fe_store(o + 10, fe_inv(a));
}



// ---- micro-benchmark kernel (tests only) -----------------------------------

/// Dependent chains of one operation per thread: `mode` 0 = fe_mul, 1 =
/// fe_sqr, 2 = fe_add, 3 = the per-point test (β·x, x + β·x, top of −(x + β·x)).
kernel void bench(device uint4* data [[buffer(0)]],
                  constant uint& iters [[buffer(1)]],
                  constant uint& mode [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    Fe a = fe_load(data + 4 * i);
    Fe b = fe_load(data + 4 * i + 2);
    if (mode == 0) {
        for (uint k = 0; k < iters; k++) {
            a = fe_mul(a, b);
        }
    } else if (mode == 1) {
        for (uint k = 0; k < iters; k++) {
            a = fe_sqr(a);
        }
    } else if (mode == 2) {
        for (uint k = 0; k < iters; k++) {
            a = fe_add(a, b);
        }
    } else if (mode == 4) {
        for (uint k = 0; k < iters; k++) {
            a = fe_mul_plain(a, b);
        }
    } else {
        for (uint k = 0; k < iters; k++) {
            Fe bx = fe_mul(BETA, a);
            Fe s = fe_add(bx, a);
            uint2 top = neg_top(s);
            a.v[0] ^= top.x;
            a.v[1] ^= top.y;
        }
    }
    fe_store(data + 4 * i, a);
}
