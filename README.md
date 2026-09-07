# spaghetti

Vanity address generator for [BIP352](https://github.com/bitcoin/bips/blob/master/bip-0352.mediawiki) silent payments. It finds a **scan secret key** whose derived addresses start with a chosen prefix, e.g. `sp1qq?pasta…`.

A v0 silent payment address is `bech32m(hrp, q ++ convertbits(ser_P(B_scan) ++ ser_P(B_m)))`. The first 52 data characters depend on `B_scan` only, so one vanity scan key gives the prefix to every address derived from it, including labeled ones: a wallet that hands out one labeled address per contact or customer shares one scan key across all of them, so a vanity scan key gives every labeled address the prefix.

## Install

```
cargo install --git https://github.com/louneskmt/spaghetti
```

or clone and `cargo build --release` (binary in `target/release/spaghetti`). `RUSTFLAGS="-C target-cpu=native"` made no measurable difference on an Apple M2; it may help on other CPUs.

## Usage

```
spaghetti pasta                 # sp1qq?pasta…  (? = any of the 8 allowed 6th chars)
spaghetti sp1qqgpasta           # fixed 6th char: forces an even-y scan key
spaghetti -n testnet pasta      # tsp1qq?pasta…
spaghetti -k 3 pasta penne      # any-of, stop after 3 matches
spaghetti -s 02…33-byte-hex pasta   # render the example address with a real spend key
spaghetti -b 02…33-byte-hex pasta   # split-key mode: seed-recoverable key (see below)
spaghetti --xpub xpub6… pasta       # same, base key from the xpub of m/352'/0'/0'/1'
spaghetti recover --address sp1qq… -b 02…   # find the tweak of a published address
spaghetti apply --scan-priv <d hex> --tweak 48213946821/1/-   # wallet scan key
```

```
spaghetti [OPTIONS] <PATTERN>...

  <PATTERN>...  address prefix, full (sp1qq?pasta) or bare (pasta = sp1qq?pasta); ? = any char
  -n, --network <NETWORK>      mainnet | testnet (hrp tsp, also used for signet/regtest) [default: mainnet]
  -c, --cores <N>              threads [default: available_parallelism]
  -k, --count <N>              stop after N matches [default: 1]
  -s, --spend-pubkey <HEX33>   spend public key used to render the example address (random throwaway one if omitted)
  -b, --base-pubkey <HEX33>    split-key mode: search offsets from this compressed scan pubkey D
      --xpub <XPUB>            split-key mode: D = child 0 (non-hardened) of this extended pubkey (xpub or tpub);
                               give the node m/352'/0'/0'/1' (testnet: m/352'/1'/0'/1'). Mutually exclusive with -b.
  -q, --quiet                  no progress output

spaghetti recover --address <SP_ADDRESS> (-b <HEX33> | --xpub <XPUB>) [-c N] [--baby-bits K]
spaghetti apply --scan-priv <HEX32> --tweak <t/e/s> [--address <SP_ADDRESS>]
```

Output per match (stdout; progress goes to stderr):

```
scan secret key : e9ddb9cd680fab7aba101d2e153ccaee3bfbcecd4f3d4f4f61b82b503e89bb65
scan public key : 0343d82fa7bdb35841776f9e0392a64b86f6b1fbd36bfe2ad2fcd8f4e6dbf14bc5
spend public key: 03edb2b32ed41a5ece06a36f32e1c8992aff392ea6c6703bc41f1986a53ccf79cb (example, random)
address         : sp1qqdpasta8hke4ssthd70q8y4xfwr0dv0m6d4lu2kjlnv0fekm799u2qldk2eja4q6tm8qdgm0xtsu3xf2luujafkxwqaug8ces6jnenmeev8jz5rx
found after 23.23 M candidates in 101 ms (229.81 M/s)
```

## Seed-recoverable keys (split-key mode)

**Why.** A random vanity scan key is not on any BIP32 path, so a seed backup does not cover it. Losing the scan key is worse than losing the ability to *see* payments: spending a silent payment output needs the ECDH shared secret, which is computed from the scan key, so every coin received on that key becomes unspendable. Split-key mode keeps the vanity key one public number away from the seed.

**How.** The wallet's standard BIP352 scan key is `d` (`m/352'/coin'/account'/1'/0`) with public key `D = d·G`. Instead of a random key, `spaghetti -b <D>` searches a small offset `t` such that

```
B = s · λ^e · (D + t·G)          has the prefix,
scan_priv = s · λ^e · (d + t)  mod n
```

where `e ∈ {0,1,2}` is the endomorphism power the search applied (`λ` is the secp256k1 GLV scalar, `λ³ = 1`) and `s = ±1` fixes the y parity when the 6th char is fixed. The result is printed as a **tweak string** `<t>/<e>/<s>`, e.g. `48213946821/1/-`. Worker `i` walks offsets `t ∈ [i·2^44, (i+1)·2^44)`, so with at most 256 threads (`-c`, threads are offset ranges and may exceed the core count) every published `t` is below `2^52`. That bound is what makes `t` recoverable by brute force (next paragraph); `2^44` per thread covers about 2^45.6 x candidates per thread (three per point), enough for prefixes up to 9 characters on a dozen threads; the search warns when the pattern needs more coverage than the thread count provides.

```
spaghetti -b 0201e79b7d70f29abcc2c41665ac88131fe0ea7be269e558f1aac4ab78522bf51f pasta

base scan pubkey  : 0201e79b7d70f29abcc2c41665ac88131fe0ea7be269e558f1aac4ab78522bf51f
tweak             : 87960930315849/1/+
vanity scan pubkey: 02c3d82fa34db5af529b72464bd2f21e1c396d21fb1348a187a76f91a51fd6c998
spend public key  : 027a1cdf30d2a8a4ff52a66ce9ed0a0dac23ae5c413b5fc5d67d9f73b6f0234dda (example, random)
address           : sp1qqtpastarfk66755mwfryh5hjrcwrjmfplvf53gv85aherfgl6myesqn6rn0np54g5nl49fnva8ks5rdvywh9csfmtlzavlvlwwm0qg6dmgn7rss5
scan_priv = λ^1·(d + 87960930315849) mod n   → run: spaghetti apply --scan-priv <d hex> --tweak 87960930315849/1/+
```

**What to store.** The tweak string, next to the descriptor. It is public data (it reveals nothing about `d`), plaintext is fine, and it is not even required: only the seed is secret, and the tweak can be recomputed.

**Recovery without the tweak.** Given the seed and any address ever published (the first 52 characters of every address of the wallet encode `B`), `spaghetti recover` finds the tweak again:

```
spaghetti recover --address sp1qqtpasta… -b <D>      # or --xpub <xpub of m/352'/0'/0'/1'>
```

For each of the six `(e, s)` variants it solves `s·λ^{-e}·B − D = t·G` for `t < 2^52` with baby-step giant-step: a table of `x(j·G)` for `j < 2^K` (`--baby-bits`, default 22, 60 MB, built in 0.4 s) and up to `2^(52−K)` giant steps per variant, walked with the same batched-inversion machinery as the search. The giant steps run in growing levels over all six variants, so a small `t` is found quickly whatever its variant. Measured on an Apple M2 Pro with 12 threads: about 200 M giant steps/s, i.e. up to 5.5 s per variant at the default `K`; the example above (`t ≈ 5·2^44`, third variant) took 12 s and the worst case (`t = 2^51 + 12345`, last variant) 30 s. Each extra baby bit halves the giant-step work and doubles the table (`--baby-bits 24`: 240 MB, ~4× faster).

**Wallet key.** `spaghetti apply --scan-priv <d hex> --tweak <t/e/s> [--address sp1qq…]` prints the vanity scan secret key (`s·λ^e·(d + t) mod n`) and its public key, and checks them against the address if given. Import that secret key as the wallet's scan key; the spend key is unchanged.

**Labels caveat.** BIP352 label tweaks are `hash(ser_256(b_scan) ‖ ser_32(m))`, so labeled spend keys, and the change label `m = 0`, depend on the scan key. Everything must be derived from the vanity scan key, and the switch has to happen before the first address is issued: addresses (and their labels) handed out under the original `d` are not detectable with the vanity key.

**Obtaining `D`.** Export the extended public key of `m/352'/0'/0'/1'` (`m/352'/1'/0'/1'` for testnet, a `tpub`) and pass it with `--xpub`: `spaghetti` derives child 0 itself (plain BIP32, HMAC-SHA512). Or pass the compressed public key of `m/352'/0'/0'/1'/0` directly with `-b`. Both `recover` and the search accept either form; the network must match (`-n` for the search, the address hrp for `recover`).

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

Expected candidates = `2^(constrained x bits)` = `32^n` for `n` fixed chars after `sp1qq?`, times 4 if the 6th char is fixed. Measured on an Apple M2 Pro (`--cores 12`): **~560 M x-candidates/s** (≈67 M/s per performance core; each x candidate covers both y parities, see below).

| pattern       | fixed chars | expected candidates | expected time at 560 M/s |
| ------------- | ----------- | ------------------- | ------------------------ |
| `pas`         | 3           | 2^15 ≈ 33 K         | 0.06 ms                  |
| `sp1qqgpas`   | 3 + parity  | 2^17 ≈ 131 K        | 0.2 ms                   |
| `pasta`       | 5           | 2^25 ≈ 33.6 M       | 0.06 s                   |
| `lasagna`     | 7           | 2^35 ≈ 34.4 G       | 1 min                    |
| `farfalle`    | 8           | 2^40 ≈ 1.10 T       | 33 min                   |
| `farfalle7`   | 9           | 2^45 ≈ 35.2 T       | 17 h                     |
| `pastasauce`  | 10          | 2^50 ≈ 1.13 P       | 23 days                  |

The search is memoryless: the ETA in the progress line is `(expected − tested) / rate`, but the true expected remaining time is always `expected / rate` regardless of how long you have already searched.

## How it works

Per thread, [VanitySearch](https://github.com/JeanLucPons/VanitySearch)-style on the CPU:

1. Random start scalar `k0`, centre point `C = k0·G` (computed once with `k256`).
2. A shared table holds `j·G` for `j = 1..=H` (`H = 1024`, `--batch` to change) and the jump `(2H+1)·G`.
3. Per batch, the x coordinates of `C ± j·G` are computed with a single field inversion (Montgomery's trick over `T[j].x − C.x`, as four interleaved product chains so consecutive multiplications do not wait on each other): 3 multiplications per inverse plus 2 multiplications and 2 squarings per pair of points. The same inversion batch also produces `C += (2H+1)·G`.
4. **x-only**: result y coordinates are never computed. Negating the scalar flips the y parity for free, so matching happens on x alone and the parity required by a fixed 6th char is fixed afterwards.
5. **Endomorphism**: for every x, `β·x` and `β²·x` are also tested (scalars `λk`, `λ²k`). One multiplication (`β·x`) and one addition (`β²·x = −x − β·x`, since `β² + β + 1 = 0`) buy two extra candidates.
6. Pattern checks compare the top 64 bits of x as a masked `u64` first and only then fall back to a byte-wise check.
7. Matches take the slow path: reconstruct the scalar with `k256`, check the derived x, fix the parity, render the address with the `bech32` crate, verify the prefix character by character and decode the address back to the scan key.

Split-key mode reuses the same walk with centre `D + k0·G`; `recover` reuses it with generator `−2^K·G` and centre `s·λ^{-e}·B − D` for the giant steps, so the giant-step rate is the walk rate minus a table lookup (an 8 MB bitmap filter in front of a bucketed sorted array of the top 64 bits of `x`).

Field arithmetic (`src/field.rs`) is a purpose-built canonical 4×64-bit implementation of `p = 2^256 − 2^32 − 977` (no `unsafe`, no assembly), written as explicit carry chains the compiler turns into add-with-carry sequences, with the rare reduction cases out of line; it is checked against a big-integer reference in the tests. The per-point visitors of the batch loop are trait implementations rather than closures so that they are inlined into it. `k256` is used only for scalar arithmetic, setup and verification. Cross-check: the BIP352 test vector address is a unit test (`src/address.rs`).

## Security notes

- The scan key **only reveals incoming payments** (it lets the holder detect outputs, not spend them), but treat it like any wallet secret: whoever has it can link every payment to you.
- A vanity scan key from the default mode is not BIP32-derived. Wallets normally derive the scan key from the seed (`m/352'/0'/0'/1'/0`), so such a key cannot be recovered from the seed phrase: back it up separately and inject it into the wallet instead of deriving it. Split-key mode (above) avoids this: the key is the derived one plus a public, recomputable tweak.
- Keys come from the OS RNG (`getrandom` via `k256`); the search walks a public additive sequence from that random start, so every found key is as unpredictable as its start point. The randomly generated spend key printed when `-s` is omitted is a throwaway used only to render an example address.

## Development

```
cargo test                                  # field reference tests, BIP352 + BIP32 vectors, search vs k256, split-key flow
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Disclaimer

This tool was written with heavy assistance from AI coding agents (Claude). The cryptography is covered by tests against a big-integer reference, the BIP352 test vectors and k256, but it has not been independently audited. Use it at your own risk and verify any key it produces before funding it.

## License

MIT
