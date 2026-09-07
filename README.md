# spaghetti

Vanity address generator for [BIP352](https://github.com/bitcoin/bips/blob/master/bip-0352.mediawiki) silent payments. It finds a **scan secret key** whose derived addresses start with a chosen prefix, e.g. `sp1qq?pasta…`.

A v0 silent payment address is `bech32m(hrp, q ++ convertbits(ser_P(B_scan) ++ ser_P(B_m)))`. The first 52 data characters depend on `B_scan` only, so one vanity scan key gives the prefix to every address derived from it, including labeled ones: a wallet that hands out one labeled address per contact or customer shares one scan key across all of them, so a vanity scan key gives every labeled address the prefix.

## Install

```
cargo install --git https://github.com/louneskmt/spaghetti
```

or clone and `cargo build --release` (binary in `target/release/spaghetti`). For an extra few percent build with `RUSTFLAGS="-C target-cpu=native"`.

## Usage

```
spaghetti pasta                 # sp1qq?pasta…  (? = any of the 8 allowed 6th chars)
spaghetti sp1qqgpasta           # fixed 6th char: forces an even-y scan key
spaghetti -n testnet pasta      # tsp1qq?pasta…
spaghetti -k 3 pasta penne      # any-of, stop after 3 matches
spaghetti -s 02…33-byte-hex pasta   # render the example address with a real spend key
```

```
spaghetti [OPTIONS] <PATTERN>...

  <PATTERN>...  address prefix, full (sp1qq?pasta) or bare (pasta = sp1qq?pasta); ? = any char
  -n, --network <NETWORK>      mainnet | testnet (hrp tsp, also used for signet/regtest) [default: mainnet]
  -c, --cores <N>              threads [default: available_parallelism]
  -k, --count <N>              stop after N matches [default: 1]
  -s, --spend-pubkey <HEX33>   spend public key used to render the example address (random throwaway one if omitted)
  -q, --quiet                  no progress output
```

Output per match (stdout; progress goes to stderr):

```
scan secret key : e9ddb9cd680fab7aba101d2e153ccaee3bfbcecd4f3d4f4f61b82b503e89bb65
scan public key : 0343d82fa7bdb35841776f9e0392a64b86f6b1fbd36bfe2ad2fcd8f4e6dbf14bc5
spend public key: 03edb2b32ed41a5ece06a36f32e1c8992aff392ea6c6703bc41f1986a53ccf79cb (example, random)
address         : sp1qqdpasta8hke4ssthd70q8y4xfwr0dv0m6d4lu2kjlnv0fekm799u2qldk2eja4q6tm8qdgm0xtsu3xf2luujafkxwqaug8ces6jnenmeev8jz5rx
found after 23.23 M candidates in 101 ms (229.81 M/s)
```

## What can be chosen

Every v0 address starts with `sp1qq` (`tsp1qq` on testnet): `sp` is the hrp, `1` the separator, the first `q` is the version, and the second `q` encodes the five zero bits at the top of the SEC1 tag byte (`0x02`/`0x03`).

The 6th character encodes the tag's last two bits (`1` + y parity) and the top two bits of the x coordinate, so only 8 values are possible:

| 6th char | y parity | x top bits |
| -------- | -------- | ---------- |
| `g`      | even     | `00`       |
| `f`      | even     | `01`       |
| `2`      | even     | `10`       |
| `t`      | even     | `11`       |
| `v`      | odd      | `00`       |
| `d`      | odd      | `01`       |
| `w`      | odd      | `10`       |
| `0`      | odd      | `11`       |

From the 7th character on, every character is 5 free bits of x. A bare pattern `pasta` means `sp1qq?pasta`; use the full form to fix the 6th char. The bech32 charset is `qpzry9x8gf2tvdw0s3jn54khce6mua7l`: `1`, `b`, `i`, `o` never appear. At most 50 characters may follow the 6th one.

## Difficulty

Expected candidates = `2^(constrained x bits)` = `32^n` for `n` fixed chars after `sp1qq?`, times 4 if the 6th char is fixed. Measured on an Apple M2 Pro (`--cores 12`): **~330 M x-candidates/s** (≈39 M/s per performance core; each x candidate covers both y parities, see below).

| pattern       | fixed chars | expected candidates | expected time at 330 M/s |
| ------------- | ----------- | ------------------- | ------------------------ |
| `pas`         | 3           | 2^15 ≈ 33 K         | 0.1 ms                   |
| `sp1qqgpas`   | 3 + parity  | 2^17 ≈ 131 K        | 0.4 ms                   |
| `pasta`       | 5           | 2^25 ≈ 33.6 M       | 0.1 s                    |
| `lasagna`     | 7           | 2^35 ≈ 34.4 G       | 1.7 min                  |
| `farfalle`    | 8           | 2^40 ≈ 1.10 T       | 56 min                   |
| `farfalle7`   | 9           | 2^45 ≈ 35.2 T       | 30 h                     |
| `pastasauce`  | 10          | 2^50 ≈ 1.13 P       | 40 days                  |

The search is memoryless: the ETA in the progress line is `(expected − tested) / rate`, but the true expected remaining time is always `expected / rate` regardless of how long you have already searched.

## How it works

Per thread, [VanitySearch](https://github.com/JeanLucPons/VanitySearch)-style on the CPU:

1. Random start scalar `k0`, centre point `C = k0·G` (computed once with `k256`).
2. A shared table holds `j·G` for `j = 1..=H` (`H = 1024`, `--batch` to change) and the jump `(2H+1)·G`.
3. Per batch, the x coordinates of `C ± j·G` are computed with a single field inversion (Montgomery's trick over `T[j].x − C.x`): 3 multiplications per inverse plus about 2 multiplications and 2 squarings per pair of points. The same inversion batch also produces `C += (2H+1)·G`.
4. **x-only**: result y coordinates are never computed. Negating the scalar flips the y parity for free, so matching happens on x alone and the parity required by a fixed 6th char is fixed afterwards.
5. **Endomorphism**: for every x, `β·x` and `β²·x` are also tested (scalars `λk`, `λ²k`). Two multiplications buy two extra candidates.
6. Pattern checks compare the top 64 bits of x as a masked `u64` first and only then fall back to a byte-wise check.
7. Matches take the slow path: reconstruct the scalar with `k256`, check the derived x, fix the parity, render the address with the `bech32` crate, verify the prefix character by character and decode the address back to the scan key.

Field arithmetic (`src/field.rs`) is a purpose-built canonical 4×64-bit implementation of `p = 2^256 − 2^32 − 977` (no `unsafe`, no assembly), checked against a big-integer reference in the tests. `k256` is used only for scalar arithmetic, setup and verification. Cross-check: the BIP352 test vector address is a unit test (`src/address.rs`).

## Security notes

- The scan key **only reveals incoming payments** (it lets the holder detect outputs, not spend them), but treat it like any wallet secret: whoever has it can link every payment to you.
- A vanity scan key is not BIP32-derived. Wallets normally derive the scan key from the seed (`m/352'/0'/0'/1'/0`), so a vanity key cannot be recovered from the seed phrase: back it up separately and inject it into the wallet instead of deriving it.
- Keys come from the OS RNG (`getrandom` via `k256`); the search walks a public additive sequence from that random start, so every found key is as unpredictable as its start point. The randomly generated spend key printed when `-s` is omitted is a throwaway used only to render an example address.

## Development

```
cargo test                                  # field reference tests, BIP352 vector, search vs k256
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## License

MIT
