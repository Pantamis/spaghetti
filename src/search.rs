//! VanitySearch-style batched search over `C ± j·G`.
//!
//! One worker owns a moving centre point `C = k0·G`. Per batch it computes the
//! x coordinates of `C + j·G` and `C - j·G` for `j = 1..=H` with a single field
//! inversion (Montgomery's trick), tests `x`, `β·x` and `β²·x` (endomorphism:
//! scalars `k`, `λk`, `λ²k`) against the patterns, then jumps `C += (2H+1)·G`
//! using the same batched inversion. Result y coordinates are never computed:
//! the y parity is fixed afterwards by negating the scalar.

use crate::field::Fe;
use crate::pattern::PatternSet;
use k256::elliptic_curve::Generate;
use k256::elliptic_curve::point::AffineCoordinates;
use k256::elliptic_curve::scalar::FromUintUnchecked;
use k256::elliptic_curve::sec1::ToSec1Point;
use k256::{NonZeroScalar, ProjectivePoint, Scalar, U256};

const LAMBDA_HEX: &str = "5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72";

/// λ such that λ·(x, y) = (β·x, y).
pub fn lambda() -> Scalar {
    Scalar::from_uint_unchecked(U256::from_be_hex(LAMBDA_HEX))
}

/// Affine x and y of a k256 point as field elements.
pub fn affine_xy(point: &ProjectivePoint) -> (Fe, Fe) {
    let affine = point.to_affine();
    let x: [u8; 32] = affine.x().into();
    let y: [u8; 32] = affine.y().into();
    (Fe::from_bytes_be(&x), Fe::from_bytes_be(&y))
}

/// Compressed SEC1 encoding of a k256 point.
pub fn compressed(point: &ProjectivePoint) -> [u8; 33] {
    let sec1 = point.to_affine().to_sec1_point(true);
    let mut out = [0u8; 33];
    out.copy_from_slice(sec1.as_bytes());
    out
}

/// Precomputed `j·G` for `j = 1..=H` plus the batch jump `(2H+1)·G`.
pub struct Table {
    pub half: usize,
    x: Vec<Fe>,
    y: Vec<Fe>,
    jump_x: Fe,
    jump_y: Fe,
}

impl Table {
    pub fn new(half: usize) -> Table {
        let half = half.max(1);
        let mut x = Vec::with_capacity(half);
        let mut y = Vec::with_capacity(half);
        let mut point = ProjectivePoint::GENERATOR;
        for _ in 0..half {
            let (px, py) = affine_xy(&point);
            x.push(px);
            y.push(py);
            point += ProjectivePoint::GENERATOR;
        }
        let jump = ProjectivePoint::GENERATOR * Scalar::from(2 * half as u64 + 1);
        let (jump_x, jump_y) = affine_xy(&jump);
        Table {
            half,
            x,
            y,
            jump_x,
            jump_y,
        }
    }

    /// Scalar distance between consecutive batch centres.
    pub fn step(&self) -> u64 {
        2 * self.half as u64 + 1
    }

    /// x candidates produced per batch (`x`, `βx`, `β²x` for `2H+1` points).
    pub fn candidates_per_batch(&self) -> u64 {
        3 * self.step()
    }
}

/// Inverts every element of `values` with one field inversion.
/// `scratch` and `out` must have the same length as `values`.
/// Returns `false` (leaving `out` unspecified) if any value is zero.
pub fn batch_invert(values: &[Fe], scratch: &mut [Fe], out: &mut [Fe]) -> bool {
    let n = values.len();
    if n == 0 {
        return true;
    }
    let scratch = &mut scratch[..n];
    let out = &mut out[..n];
    scratch[0] = values[0];
    for i in 1..n {
        scratch[i] = scratch[i - 1].mul(&values[i]);
    }
    if scratch[n - 1].is_zero() {
        return false;
    }
    let mut inv = scratch[n - 1].invert();
    for i in (1..n).rev() {
        out[i] = inv.mul(&scratch[i - 1]);
        inv = inv.mul(&values[i]);
    }
    out[0] = inv;
    true
}

/// A pattern hit before scalar reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Batch start scalar.
    pub k0: Scalar,
    /// `k = k0 + offset` before the endomorphism.
    pub offset: i64,
    /// Power of λ applied (0, 1 or 2).
    pub endo: u8,
    /// x coordinate of `λ^endo · (k0 + offset) · G`.
    pub x: Fe,
    pub pattern: usize,
}

/// A verified vanity scan key.
#[derive(Clone, Debug)]
pub struct Found {
    pub secret: [u8; 32],
    pub pubkey: [u8; 33],
    pub pattern: usize,
}

pub struct Worker<'a> {
    table: &'a Table,
    k0: Scalar,
    cx: Fe,
    cy: Fe,
    dx: Vec<Fe>,
    scratch: Vec<Fe>,
    inv: Vec<Fe>,
}

impl<'a> Worker<'a> {
    /// Worker starting at a random scalar.
    pub fn new(table: &'a Table) -> Result<Worker<'a>, String> {
        let start = NonZeroScalar::try_generate().map_err(|e| format!("rng failure: {e}"))?;
        Ok(Self::with_start(table, *start.as_ref()))
    }

    pub fn with_start(table: &'a Table, k0: Scalar) -> Worker<'a> {
        let (cx, cy) = affine_xy(&(ProjectivePoint::GENERATOR * k0));
        let n = table.half + 1;
        Worker {
            table,
            k0,
            cx,
            cy,
            dx: vec![Fe::ZERO; n],
            scratch: vec![Fe::ZERO; n],
            inv: vec![Fe::ZERO; n],
        }
    }

    /// Runs one batch, calling `visit(offset, x)` for `x((k0 + offset)·G)` with
    /// `offset` in `-H..=H`, then moves to the next batch. Returns `false` if
    /// the batch was skipped because of a degenerate (zero) difference.
    #[inline]
    pub fn batch<F: FnMut(i64, &Fe)>(&mut self, mut visit: F) -> bool {
        let table = self.table;
        let h = table.half;
        let (cx, cy) = (self.cx, self.cy);
        let xs = &table.x[..h];
        let ys = &table.y[..h];
        {
            let (dx, dx_jump) = self.dx.split_at_mut(h);
            for (d, tx) in dx.iter_mut().zip(xs) {
                *d = tx.sub(&cx);
            }
            dx_jump[0] = table.jump_x.sub(&cx);
        }
        if !batch_invert(&self.dx, &mut self.scratch, &mut self.inv) {
            // C == ±j·G for some j in the table (probability ~2^-247): jump
            // via k256 instead and skip this batch.
            self.k0 = self.k0.add(&Scalar::from(table.step()));
            let (cx, cy) = affine_xy(&(ProjectivePoint::GENERATOR * self.k0));
            self.cx = cx;
            self.cy = cy;
            return false;
        }
        visit(0, &cx);
        let neg_cy = cy.neg();
        for (j, ((tx, ty), inv)) in xs.iter().zip(ys).zip(&self.inv[..h]).enumerate() {
            let offset = j as i64 + 1;
            let lambda_plus = ty.sub(&cy).mul(inv);
            let x_plus = lambda_plus.square().sub(&cx).sub(tx);
            visit(offset, &x_plus);
            let lambda_minus = neg_cy.sub(ty).mul(inv);
            let x_minus = lambda_minus.square().sub(&cx).sub(tx);
            visit(-offset, &x_minus);
        }
        // Jump: C += (2H+1)·G.
        let lambda = table.jump_y.sub(&cy).mul(&self.inv[h]);
        let nx = lambda.square().sub(&cx).sub(&table.jump_x);
        let ny = lambda.mul(&cx.sub(&nx)).sub(&cy);
        self.cx = nx;
        self.cy = ny;
        self.k0 = self.k0.add(&Scalar::from(table.step()));
        true
    }

    /// One batch of pattern matching; pushes hits into `out`.
    #[inline]
    pub fn search_batch(&mut self, patterns: &PatternSet, out: &mut Vec<Candidate>) -> bool {
        let k0 = self.k0;
        self.batch(|offset, x| {
            let bx = Fe::BETA.mul(x);
            let b2x = Fe::BETA2.mul(x);
            let mut hit = |endo: u8, candidate: &Fe| {
                if let Some(pattern) = patterns.find(candidate) {
                    out.push(Candidate {
                        k0,
                        offset,
                        endo,
                        x: *candidate,
                        pattern,
                    });
                }
            };
            hit(0, x);
            hit(1, &bx);
            hit(2, &b2x);
        })
    }
}

/// Reconstructs the scalar for a candidate, checks it against k256, and fixes
/// the y parity when the pattern requires one.
pub fn resolve(candidate: &Candidate, patterns: &PatternSet) -> Result<Found, String> {
    let magnitude = Scalar::from(candidate.offset.unsigned_abs());
    let mut k = if candidate.offset >= 0 {
        candidate.k0.add(&magnitude)
    } else {
        candidate.k0.sub(&magnitude)
    };
    for _ in 0..candidate.endo {
        k = k.mul(&lambda());
    }
    let pattern = patterns
        .patterns
        .get(candidate.pattern)
        .ok_or_else(|| "internal: pattern index out of range".to_string())?;
    let mut point = ProjectivePoint::GENERATOR * k;
    let (x, _) = affine_xy(&point);
    if x != candidate.x {
        return Err("internal: reconstructed key does not reproduce the candidate x".to_string());
    }
    let mut pubkey = compressed(&point);
    if let Some(want_odd) = pattern.parity
        && (pubkey[0] == 0x03) != want_odd
    {
        k = k.negate();
        point = ProjectivePoint::GENERATOR * k;
        pubkey = compressed(&point);
    }
    if !pattern.matches_bytes(&pubkey[1..].try_into().map_err(|_| "internal: bad pubkey")?) {
        return Err("internal: pubkey x does not match the pattern".to_string());
    }
    Ok(Found {
        secret: k.to_bytes().into(),
        pubkey,
        pattern: candidate.pattern,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address;
    use crate::pattern::Network;
    use k256::elliptic_curve::PrimeField;

    fn random_scalar() -> Scalar {
        *NonZeroScalar::try_generate().unwrap().as_ref()
    }

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    #[test]
    fn batch_invert_matches_individual_inverts() {
        let values: Vec<Fe> = (0..37).map(|_| x_of(&random_scalar())).collect();
        let mut scratch = vec![Fe::ZERO; values.len()];
        let mut out = vec![Fe::ZERO; values.len()];
        assert!(batch_invert(&values, &mut scratch, &mut out));
        for (v, inv) in values.iter().zip(&out) {
            assert_eq!(*inv, v.invert());
            assert_eq!(v.mul(inv), Fe::ONE);
        }
        let mut with_zero = values.clone();
        with_zero[5] = Fe::ZERO;
        assert!(!batch_invert(&with_zero, &mut scratch, &mut out));
        assert!(batch_invert(&[], &mut [], &mut []));
    }

    #[test]
    fn batch_x_coordinates_match_k256() {
        let table = Table::new(256);
        let k0 = random_scalar();
        let mut worker = Worker::with_start(&table, k0);
        let mut seen = vec![None; 2 * table.half + 1];
        assert!(worker.batch(|offset, x| {
            let idx = (offset + table.half as i64) as usize;
            assert!(seen[idx].is_none(), "offset {offset} visited twice");
            seen[idx] = Some(*x);
        }));
        for (idx, x) in seen.iter().enumerate() {
            let offset = idx as i64 - table.half as i64;
            let magnitude = Scalar::from(offset.unsigned_abs());
            let k = if offset >= 0 {
                k0.add(&magnitude)
            } else {
                k0.sub(&magnitude)
            };
            assert_eq!(x.unwrap(), x_of(&k), "offset {offset}");
        }
        // After the batch the centre moved by 2H+1.
        let next = k0.add(&Scalar::from(table.step()));
        assert_eq!(worker.k0, next);
        assert_eq!(
            affine_xy(&(ProjectivePoint::GENERATOR * next)),
            (worker.cx, worker.cy)
        );
        // And a second batch is consistent too.
        let mut count = 0;
        assert!(worker.batch(|offset, x| {
            count += 1;
            let k = if offset >= 0 {
                next.add(&Scalar::from(offset.unsigned_abs()))
            } else {
                next.sub(&Scalar::from(offset.unsigned_abs()))
            };
            assert_eq!(*x, x_of(&k));
        }));
        assert_eq!(count, 2 * table.half + 1);
    }

    #[test]
    fn endomorphism_scalar_relation() {
        for _ in 0..8 {
            let k = random_scalar();
            let x = x_of(&k);
            let lk = k.mul(&lambda());
            let l2k = lk.mul(&lambda());
            assert_eq!(x_of(&lk), Fe::BETA.mul(&x));
            assert_eq!(x_of(&l2k), Fe::BETA2.mul(&x));
            assert_eq!(x_of(&l2k.mul(&lambda())), x);
        }
    }

    #[test]
    fn small_table_and_step() {
        let table = Table::new(4);
        assert_eq!(table.step(), 9);
        assert_eq!(table.candidates_per_batch(), 27);
        assert_eq!(Table::new(0).half, 1);
    }

    /// End-to-end: search a short pattern, then independently recompute the
    /// address from the secret key with k256 + bech32 and decode it.
    #[test]
    fn search_and_verify_independently() {
        let table = Table::new(64);
        let spend = [2u8; 33];
        for (input, network, hrp) in [
            ("sp1qq?q", Network::Mainnet, address::HRP_MAINNET),
            ("sp1qqgp", Network::Mainnet, address::HRP_MAINNET),
            ("sp1qqvz", Network::Mainnet, address::HRP_MAINNET),
            ("tsp1qq?r", Network::Testnet, address::HRP_TESTNET),
        ] {
            let patterns = PatternSet::parse(&[input.to_string()], network).unwrap();
            let mut worker = Worker::new(&table).unwrap();
            let mut hits = Vec::new();
            let mut batches = 0;
            while hits.is_empty() {
                worker.search_batch(&patterns, &mut hits);
                batches += 1;
                assert!(batches < 10_000, "no hit for {input}");
            }
            for candidate in &hits {
                let found = resolve(candidate, &patterns).unwrap();
                // Independent path: scalar → point → compressed → bech32m.
                let scalar = Scalar::from_repr_vartime(found.secret.into()).unwrap();
                let point = ProjectivePoint::GENERATOR * scalar;
                let pubkey = compressed(&point);
                assert_eq!(pubkey, found.pubkey);
                let addr = address::encode(hrp, &pubkey, &spend);
                assert!(
                    patterns.patterns[found.pattern].matches_address(&addr),
                    "{input}: {addr}"
                );
                let (got_hrp, version, payload) = address::decode(&addr).unwrap();
                assert_eq!(got_hrp, hrp);
                assert_eq!(version, bech32::Fe32::Q);
                assert_eq!(&payload[..33], &pubkey);
            }
        }
    }

    #[test]
    fn resolve_rejects_wrong_x() {
        let table = Table::new(4);
        let patterns = PatternSet::parse(&["sp1qq".to_string()], Network::Mainnet).unwrap();
        let worker = Worker::new(&table).unwrap();
        let candidate = Candidate {
            k0: worker.k0,
            offset: 1,
            endo: 0,
            x: Fe::ONE,
            pattern: 0,
        };
        assert!(resolve(&candidate, &patterns).is_err());
    }
}
