// secp256k1 x-only batched walk on NVIDIA GPUs: the CUDA port of the Metal
// kernel (src/gpu/search.metal), itself the GPU form of `Walk::batch` and
// `Searcher` (src/search.rs). One thread owns one walk; per batch it computes
// the x coordinates of `C ± j·G` for `j = 1..=H` with a single field
// inversion (Montgomery's trick, prefix products kept in a per-thread column
// of `scratch`), tests `x`, `β·x` and `β²·x` of every point against the
// pattern keys (top 64 bits of x), appends hits to `hits`, and finally jumps
// `C += (2H+1)·G`. The host owns the scalars: a thread's centre after `b`
// batches is `base + (k0[t] + b·(2H+1))·G`, so a hit record needs only the
// thread, the batch index within the launch, the signed offset and the
// endomorphism power.
//
// The host compiles this file with NVRTC at start-up and prepends:
//   #define HALF <H>         half-batch, a compile-time loop bound
//   #define BLOCK <threads>  threads per block (launch bounds)
//   #define MINB <blocks>    minimum resident blocks per SM (launch bounds)
//   #define KEY_TEST(hi, lo) <the pattern keys as literal mask/value tests>
//
// Field arithmetic: canonical (`< p`) 8×32-bit little-endian limbs with PTX
// carry chains (`mad.lo.cc` / `madc.hi.cc`, one IMAD per half product): a
// row-wise schoolbook product, a dedicated squaring (28 cross products,
// doubled, plus the diagonal), and the reduction of `reduce_wide` in
// src/field.rs (two folds by `c = 2^256 mod p = 2^32 + 977`, one conditional
// subtraction of `p`). The carry flag lives across consecutive `asm volatile`
// statements; the compiler emits no carry-using instruction of its own in
// between (the pattern of VanitySearch and BitCrack). The tests compare every
// operation against the CPU implementation.

typedef unsigned int u32;
typedef unsigned long long u64;

#ifndef HALF
#define HALF 8
#endif
#ifndef BLOCK
#define BLOCK 128
#endif
#ifndef MINB
#define MINB 1
#endif
#ifndef KEY_TEST
#define KEY_TEST(hi, lo) false
#endif

struct Fe {
    u32 v[8];
};

/// `c = 2^256 mod p = 2^32 + 977`: 977 at limb 0, 1 at limb 1.
#define C_LO 977u

// ---- PTX carry-chain primitives ----------------------------------------------

#define ADD_CC(r, a, b) asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define ADDC_CC(r, a, b) asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define ADDC(r, a, b) asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define SUB_CC(r, a, b) asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define SUBC_CC(r, a, b) asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define SUBC(r, a, b) asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b))
#define MADLO_CC(r, a, b, c) asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define MADLOC_CC(r, a, b, c) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define MADHI(r, a, b, c) asm volatile("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define MADHI_CC(r, a, b, c) asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define MADHIC_CC(r, a, b, c) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define MADHIC(r, a, b, c) asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))

// ---- field arithmetic -------------------------------------------------------

__device__ __forceinline__ Fe fe_load(const uint4* p) {
    uint4 a = p[0], b = p[1];
    Fe r = {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
    return r;
}

/// Read-only (table) load through the non-coherent cache: every thread of a
/// warp reads the same entry, which the cache broadcasts.
__device__ __forceinline__ Fe fe_ldg(const uint4* __restrict__ p) {
    uint4 a = __ldg(p), b = __ldg(p + 1);
    Fe r = {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
    return r;
}

__device__ __forceinline__ void fe_store(uint4* p, const Fe& f) {
    p[0] = make_uint4(f.v[0], f.v[1], f.v[2], f.v[3]);
    p[1] = make_uint4(f.v[4], f.v[5], f.v[6], f.v[7]);
}

__device__ __forceinline__ bool fe_is_zero(const Fe& a) {
    u32 acc = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        acc |= a.v[i];
    }
    return acc == 0;
}

/// `a + b mod p`.
__device__ __forceinline__ Fe fe_add(const Fe& a, const Fe& b) {
    Fe r, s;
    u32 carry, cy;
    ADD_CC(r.v[0], a.v[0], b.v[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        ADDC_CC(r.v[i], a.v[i], b.v[i]);
    }
    ADDC(carry, 0u, 0u);
    // r >= p exactly when r + c carries out of 256 bits, and then r − p is
    // r + c mod 2^256; when the sum itself carried, r + c is the answer too.
    ADD_CC(s.v[0], r.v[0], C_LO);
    ADDC_CC(s.v[1], r.v[1], 1u);
#pragma unroll
    for (int i = 2; i < 8; i++) {
        ADDC_CC(s.v[i], r.v[i], 0u);
    }
    ADDC(cy, 0u, 0u);
    bool fix = (carry | cy) != 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r.v[i] = fix ? s.v[i] : r.v[i];
    }
    return r;
}

/// `a − b mod p`.
__device__ __forceinline__ Fe fe_sub(const Fe& a, const Fe& b) {
    Fe r;
    u32 borrow;
    SUB_CC(r.v[0], a.v[0], b.v[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        SUBC_CC(r.v[i], a.v[i], b.v[i]);
    }
    SUBC(borrow, 0u, 0u); // 0 or all ones
    // Wrapped below zero: add p back, i.e. subtract c from the wrapped value
    // (which is > c, so this cannot underflow).
    SUB_CC(r.v[0], r.v[0], borrow & C_LO);
    SUBC_CC(r.v[1], r.v[1], borrow & 1u);
#pragma unroll
    for (int i = 2; i < 7; i++) {
        SUBC_CC(r.v[i], r.v[i], 0u);
    }
    SUBC(r.v[7], r.v[7], 0u);
    return r;
}

__device__ __forceinline__ Fe fe_neg(const Fe& a) {
    Fe zero = {{0, 0, 0, 0, 0, 0, 0, 0}};
    return fe_sub(zero, a);
}

/// Reduces the 512-bit product `w` to a canonical element: two folds by `c`
/// and one conditional subtraction of `p` (see `reduce_wide` in field.rs).
__device__ __forceinline__ Fe fe_reduce(const u32* w) {
    u32 r[10];
    // First fold: r = lo + hi·977 + (hi << 32), 290 bits: limbs 0..=9 with
    // r[9] <= 1.
    MADLO_CC(r[0], w[8], C_LO, w[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        MADLOC_CC(r[i], w[8 + i], C_LO, w[i]);
    }
    ADDC(r[8], 0u, 0u);
    MADHI_CC(r[1], w[8], C_LO, r[1]);
#pragma unroll
    for (int i = 1; i < 7; i++) {
        MADHIC_CC(r[i + 1], w[8 + i], C_LO, r[i + 1]);
    }
    MADHIC(r[8], w[15], C_LO, r[8]);
    ADD_CC(r[1], r[1], w[8]);
#pragma unroll
    for (int i = 2; i < 9; i++) {
        ADDC_CC(r[i], r[i], w[7 + i]);
    }
    ADDC(r[9], 0u, 0u);
    // Second fold of t = r[8] + r[9]·2^32 (< 2^33): add t·977 + (t << 32),
    // i.e. limb 0 += low(r8·977), limb 1 += high(r8·977) + r9·977 + r8,
    // limb 2 += r9 and the carry of limb 1; it carries out of 256 bits at
    // most once (c2), and then the result is tiny.
    u32 t0 = r[8] * C_LO;
    u32 t1 = __umulhi(r[8], C_LO) + r[9] * C_LO;
    u32 t2;
    ADD_CC(t1, t1, r[8]);
    ADDC(t2, r[9], 0u);
    Fe out, s;
    u32 c2, cy;
    ADD_CC(out.v[0], r[0], t0);
    ADDC_CC(out.v[1], r[1], t1);
    ADDC_CC(out.v[2], r[2], t2);
#pragma unroll
    for (int i = 3; i < 8; i++) {
        ADDC_CC(out.v[i], r[i], 0u);
    }
    ADDC(c2, 0u, 0u);
    // Canonical: s = out + c carries exactly when out >= p (then out − p is
    // s mod 2^256); when the second fold carried, out is tiny and s is the
    // answer too.
    ADD_CC(s.v[0], out.v[0], C_LO);
    ADDC_CC(s.v[1], out.v[1], 1u);
#pragma unroll
    for (int i = 2; i < 8; i++) {
        ADDC_CC(s.v[i], out.v[i], 0u);
    }
    ADDC(cy, 0u, 0u);
    bool fix = (c2 | cy) != 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        out.v[i] = fix ? s.v[i] : out.v[i];
    }
    return out;
}

/// `a·b`: row-wise schoolbook, each row a low-half chain into limbs
/// `i..i+7` (carry into `i+8`) and a high-half chain into `i+1..i+8`; the
/// partial sum after row `i` is below `2^(32·(i+9))`, so the last high
/// product never carries out.
__device__ __forceinline__ Fe fe_mul(const Fe& a, const Fe& b) {
    u32 w[16];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        w[j] = a.v[0] * b.v[j];
    }
    MADHI_CC(w[1], a.v[0], b.v[0], w[1]);
#pragma unroll
    for (int j = 1; j < 7; j++) {
        MADHIC_CC(w[j + 1], a.v[0], b.v[j], w[j + 1]);
    }
    MADHIC(w[8], a.v[0], b.v[7], 0u);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        MADLO_CC(w[i], a.v[i], b.v[0], w[i]);
#pragma unroll
        for (int j = 1; j < 8; j++) {
            MADLOC_CC(w[i + j], a.v[i], b.v[j], w[i + j]);
        }
        ADDC(w[i + 8], 0u, 0u);
        MADHI_CC(w[i + 1], a.v[i], b.v[0], w[i + 1]);
#pragma unroll
        for (int j = 1; j < 7; j++) {
            MADHIC_CC(w[i + 1 + j], a.v[i], b.v[j], w[i + 1 + j]);
        }
        MADHIC(w[i + 8], a.v[i], b.v[7], w[i + 8]);
    }
    return fe_reduce(w);
}

/// `a²`: the 28 cross products `a_i·a_j` (`i < j`) row by row as in
/// `fe_mul`, doubled with a funnel shift, plus the 8 squares on the diagonal.
__device__ __forceinline__ Fe fe_sqr(const Fe& a) {
    u32 w[16];
    w[0] = 0;
    w[15] = 0;
    // Row 0: a0·a_j at limb j.
#pragma unroll
    for (int j = 1; j < 8; j++) {
        w[j] = a.v[0] * a.v[j];
    }
    MADHI_CC(w[2], a.v[0], a.v[1], w[2]);
#pragma unroll
    for (int j = 2; j < 7; j++) {
        MADHIC_CC(w[j + 1], a.v[0], a.v[j], w[j + 1]);
    }
    MADHIC(w[8], a.v[0], a.v[7], 0u);
    // Rows 1..=5: a_i·a_j (j > i) at limb i + j.
#pragma unroll
    for (int i = 1; i < 6; i++) {
        MADLO_CC(w[2 * i + 1], a.v[i], a.v[i + 1], w[2 * i + 1]);
#pragma unroll
        for (int j = i + 2; j < 8; j++) {
            MADLOC_CC(w[i + j], a.v[i], a.v[j], w[i + j]);
        }
        ADDC(w[i + 8], 0u, 0u);
        MADHI_CC(w[2 * i + 2], a.v[i], a.v[i + 1], w[2 * i + 2]);
#pragma unroll
        for (int j = i + 2; j < 7; j++) {
            MADHIC_CC(w[i + j + 1], a.v[i], a.v[j], w[i + j + 1]);
        }
        MADHIC(w[i + 8], a.v[i], a.v[7], w[i + 8]);
    }
    // Row 6: the single product a6·a7 at limb 13.
    MADLO_CC(w[13], a.v[6], a.v[7], w[13]);
    ADDC(w[14], 0u, 0u);
    MADHI(w[14], a.v[6], a.v[7], w[14]);
    // Double: the cross sum is below 2^511.
    w[15] = w[14] >> 31;
#pragma unroll
    for (int k = 14; k > 0; k--) {
        w[k] = __funnelshift_l(w[k - 1], w[k], 1);
    }
    // Diagonal (w[0] is 0 after the shift).
    MADLO_CC(w[0], a.v[0], a.v[0], w[0]);
    MADHIC_CC(w[1], a.v[0], a.v[0], w[1]);
#pragma unroll
    for (int i = 1; i < 7; i++) {
        MADLOC_CC(w[2 * i], a.v[i], a.v[i], w[2 * i]);
        MADHIC_CC(w[2 * i + 1], a.v[i], a.v[i], w[2 * i + 1]);
    }
    MADLOC_CC(w[14], a.v[7], a.v[7], w[14]);
    MADHIC(w[15], a.v[7], a.v[7], w[15]);
    return fe_reduce(w);
}

__device__ __forceinline__ Fe fe_sqn(Fe a, int n) {
    for (int i = 0; i < n; i++) {
        a = fe_sqr(a);
    }
    return a;
}

/// `a^(p−2)` with libsecp256k1's addition chain (`Fe::invert` in field.rs).
__device__ __noinline__ Fe fe_inv(Fe x) {
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

/// β (cube root of unity in the field): λ·(x, y) = (β·x, y).
__device__ __forceinline__ Fe fe_beta() {
    Fe b = {{0x719501eeu, 0xc1396c28u, 0x12f58995u, 0x9cf04975u, 0xac3434e9u, 0x6e64479eu, 0x657c0710u,
             0x7ae96a2bu}};
    return b;
}

// ---- pattern test -----------------------------------------------------------

/// Top 64 bits of `p − s` for non-zero `s` (the top of `β²·x = −(x + β·x)`):
/// `p − s` borrows into the top limbs exactly when the low 192 bits of `s`
/// exceed those of `p`, i.e. limbs 1..=5 all ones and limb 0 above `P0`.
__device__ __forceinline__ void neg_top(const Fe& s, u32& hi, u32& lo) {
    u32 ones = s.v[1] & s.v[2] & s.v[3] & s.v[4] & s.v[5];
    u32 borrow = (ones == 0xFFFFFFFFu && s.v[0] > 0xFFFFFC2Fu) ? 1u : 0u;
    lo = ~s.v[6] - borrow;
    hi = ~s.v[7] - ((borrow != 0 && s.v[6] == 0xFFFFFFFFu) ? 1u : 0u);
}

/// Hit record: 12 words `[walk, batch, offset (i32), endo, x[8]]`.
#define HIT_WORDS 12

__device__ __noinline__ void record(u32* hits, u32* counters, u32 cap, u32 t, u32 b, int offset, u32 endo,
                                    Fe x) {
    u32 idx = atomicAdd(&counters[0], 1u);
    if (idx < cap) {
        u32* h = hits + (size_t)idx * HIT_WORDS;
        h[0] = t;
        h[1] = b;
        h[2] = (u32)offset;
        h[3] = endo;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            h[4 + i] = x.v[i];
        }
    }
}

/// Tests `x`, `β·x`, `β²·x` of one point; records the hits (out of line:
/// they are rare). Returns whether anything hit.
__device__ __forceinline__ bool test_point(const Fe& x, int offset, u32 t, u32 b, u32* hits, u32* counters,
                                           u32 cap) {
    Fe bx = fe_mul(fe_beta(), x);
    Fe s = fe_add(bx, x);
    u32 hi2, lo2;
    neg_top(s, hi2, lo2);
    bool h0 = KEY_TEST(x.v[7], x.v[6]);
    bool h1 = KEY_TEST(bx.v[7], bx.v[6]);
    bool h2 = KEY_TEST(hi2, lo2);
    if (h0 | h1 | h2) {
        if (h0) record(hits, counters, cap, t, b, offset, 0, x);
        if (h1) record(hits, counters, cap, t, b, offset, 1, bx);
        if (h2) record(hits, counters, cap, t, b, offset, 2, fe_neg(s));
        return true;
    }
    return false;
}

// ---- the walk ---------------------------------------------------------------

/// `table`: `H+1` entries of `x[8], y[8]` (`j·G` for `j = 1..=H`, then the
/// jump `(2H+1)·G`). `scratch`: `H·n` elements in two planes of `uint4`
/// (low and high halves), entry `(j, t)` at `j·n + t` so a warp's accesses
/// are contiguous. `centres`: `n` entries of `x[8], y[8]`. `counters`:
/// `[hits appended, degenerate threads]`. `remaining[t]`: batches this thread
/// may still run; `flags[t]`: set to 1 when the thread hit a zero difference
/// (the host recomputes its centre).
extern "C" __global__ void __launch_bounds__(BLOCK, MINB)
    search(uint4* centres, const uint4* __restrict__ table, uint4* scratch, u32 n, u32 batches, u32 hit_cap,
           u32 first_only, u32* hits, u32* counters, u32* remaining, u32* flags) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    u32 rem = remaining[t];
    if (rem == 0) {
        return;
    }
    uint4* lo_plane = scratch;
    uint4* hi_plane = scratch + (size_t)HALF * n;
    Fe cx = fe_load(centres + 4 * (size_t)t);
    Fe cy = fe_load(centres + 4 * (size_t)t + 2);
    const Fe jump_x = fe_ldg(table + 4 * HALF);
    const u32 todo = min(rem, batches);
    for (u32 b = 0; b < todo; b++) {
        // Forward pass over the differences, the jump first so that the
        // backward pass reaches its inverse last: scratch[0] = jump.x − cx,
        // scratch[j] = scratch[j−1] · (T[j−1].x − cx) for j = 1..H−1, and
        // acc the full product (never stored: only the backward pass reads).
        Fe acc = fe_sub(jump_x, cx);
        for (u32 j = 1; j <= HALF; j++) {
            size_t at = (size_t)(j - 1) * n + t;
            lo_plane[at] = make_uint4(acc.v[0], acc.v[1], acc.v[2], acc.v[3]);
            hi_plane[at] = make_uint4(acc.v[4], acc.v[5], acc.v[6], acc.v[7]);
            acc = fe_mul(acc, fe_sub(fe_ldg(table + 4 * (j - 1)), cx));
        }
        if (fe_is_zero(acc)) {
            // C == ±j·G for some table j: the batch formulas do not apply.
            flags[t] = 1;
            atomicAdd(&counters[1], 1u);
            break;
        }
        Fe inv = fe_inv(acc);
        bool hit = test_point(cx, 0, t, b, hits, counters, hit_cap);
        // Backward pass: element j (table point j−1) gets inv · scratch[j−1].
        for (u32 j = HALF; j > 0; j--) {
            Fe tx = fe_ldg(table + 4 * (j - 1));
            Fe ty = fe_ldg(table + 4 * (j - 1) + 2);
            size_t at = (size_t)(j - 1) * n + t;
            uint4 l = lo_plane[at], h = hi_plane[at];
            Fe prefix = {{l.x, l.y, l.z, l.w, h.x, h.y, h.z, h.w}};
            Fe inv_here = fe_mul(inv, prefix);
            inv = fe_mul(inv, fe_sub(tx, cx));
            // x(C ± T) = λ² − cx − tx with λ = (±ty − cy) / (tx − cx).
            Fe sum = fe_add(cx, tx);
            Fe lambda_plus = fe_mul(fe_sub(ty, cy), inv_here);
            Fe lambda_minus = fe_mul(fe_add(ty, cy), inv_here);
            Fe x_plus = fe_sub(fe_sqr(lambda_plus), sum);
            Fe x_minus = fe_sub(fe_sqr(lambda_minus), sum);
            hit |= test_point(x_plus, (int)j, t, b, hits, counters, hit_cap);
            hit |= test_point(x_minus, -(int)j, t, b, hits, counters, hit_cap);
        }
        // Jump: C += (2H+1)·G, with inv = 1 / (jump.x − cx).
        Fe jump_y = fe_ldg(table + 4 * HALF + 2);
        Fe lambda = fe_mul(fe_sub(jump_y, cy), inv);
        Fe nx = fe_sub(fe_sub(fe_sqr(lambda), cx), jump_x);
        Fe ny = fe_sub(fe_mul(lambda, fe_sub(cx, nx)), cy);
        cx = nx;
        cy = ny;
        rem--;
        if (hit && first_only != 0) {
            // Random mode: report this batch's hits only; the host reseeds.
            break;
        }
    }
    fe_store(centres + 4 * (size_t)t, cx);
    fe_store(centres + 4 * (size_t)t + 2, cy);
    remaining[t] = rem;
}

/// Moves every centre by the same point: `C += Δ` with `delta = [Δx, Δy]`.
/// Split mode uses it to carry all walks from one range to the next
/// (`Δ = (next − current)·2^36 − batches·(2H+1)`, the same for every walk)
/// without a scalar multiplication per walk on the host. A centre with the
/// x of `Δ` (`C = ±Δ`) is left alone and flagged; the host recomputes it.
extern "C" __global__ void shift(uint4* centres, const uint4* __restrict__ delta, u32 n, u32* flags,
                                 u32* counters) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    Fe cx = fe_load(centres + 4 * (size_t)t);
    Fe cy = fe_load(centres + 4 * (size_t)t + 2);
    Fe dx = fe_ldg(delta);
    Fe dy = fe_ldg(delta + 2);
    Fe d = fe_sub(dx, cx);
    if (fe_is_zero(d)) {
        flags[t] = 1;
        atomicAdd(&counters[1], 1u);
        return;
    }
    Fe lambda = fe_mul(fe_sub(dy, cy), fe_inv(d));
    Fe nx = fe_sub(fe_sub(fe_sqr(lambda), cx), dx);
    Fe ny = fe_sub(fe_mul(lambda, fe_sub(cx, nx)), cy);
    fe_store(centres + 4 * (size_t)t, nx);
    fe_store(centres + 4 * (size_t)t + 2, ny);
}

// ---- self-test kernel -------------------------------------------------------

/// `out[i] = [a·b, a², a+b, a−b, −a, a⁻¹]` for `in[i] = [a, b]`; the tests
/// compare it with the CPU field.
extern "C" __global__ void field_test(const uint4* in, uint4* out, u32 count) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    Fe a = fe_load(in + 4 * i);
    Fe b = fe_load(in + 4 * i + 2);
    uint4* o = out + 12 * i;
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
extern "C" __global__ void bench(uint4* data, u32 iters, u32 mode) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    Fe a = fe_load(data + 4 * i);
    Fe b = fe_load(data + 4 * i + 2);
    if (mode == 0) {
        for (u32 k = 0; k < iters; k++) {
            a = fe_mul(a, b);
        }
    } else if (mode == 1) {
        for (u32 k = 0; k < iters; k++) {
            a = fe_sqr(a);
        }
    } else if (mode == 2) {
        for (u32 k = 0; k < iters; k++) {
            a = fe_add(a, b);
        }
    } else {
        for (u32 k = 0; k < iters; k++) {
            Fe bx = fe_mul(fe_beta(), a);
            Fe s = fe_add(bx, a);
            u32 hi, lo;
            neg_top(s, hi, lo);
            a.v[0] ^= hi;
            a.v[1] ^= lo;
        }
    }
    fe_store(data + 4 * i, a);
}
