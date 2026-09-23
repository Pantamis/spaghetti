//! Minimal BIP32: parse a serialised extended *public* key and derive one
//! non-hardened child. Enough to turn the xpub of `m/352'/coin'/account'/1'`
//! into the scan public key at `…/1'/0`; [`scan_account_pubkey`] also checks
//! that the xpub sits at that node (depth 4, child `1'`).

use hmac::{Hmac, KeyInit, Mac};
use k256::elliptic_curve::Group;
use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::sec1::ToSec1Point;
use k256::{ProjectivePoint, PublicKey, Scalar};
use sha2::{Digest, Sha256, Sha512};

use crate::address::Network;

const VERSION_XPUB: [u8; 4] = [0x04, 0x88, 0xB2, 0x1E];
const VERSION_TPUB: [u8; 4] = [0x04, 0x35, 0x87, 0xCF];
const VERSION_XPRV: [u8; 4] = [0x04, 0x88, 0xAD, 0xE4];
const VERSION_TPRV: [u8; 4] = [0x04, 0x35, 0x83, 0x94];

/// The version of an extended public key: `xpub` on mainnet, `tpub` on every
/// test network (testnet, signet and regtest share it and coin type `1'`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Version {
    Xpub,
    Tpub,
}

impl Version {
    /// The version wallets of `network` export.
    pub fn of(network: Network) -> Version {
        if network.is_test() {
            Version::Tpub
        } else {
            Version::Xpub
        }
    }

    /// For messages.
    pub fn describe(self) -> &'static str {
        match self {
            Version::Xpub => "an xpub (mainnet)",
            Version::Tpub => "a tpub (testnet, signet or regtest)",
        }
    }
}

/// Depth of `m/352'/coin'/account'/1'`.
const SCAN_ACCOUNT_DEPTH: u8 = 4;
/// Child number of the last step, `1'`.
const SCAN_ACCOUNT_CHILD: u32 = 0x8000_0001;

/// The parts of a serialised extended public key that derivation needs.
#[derive(Debug)]
struct Xpub {
    version: Version,
    depth: u8,
    child: u32,
    chain_code: [u8; 32],
    key: ProjectivePoint,
}

/// Base58check-decodes and parses an `xpub`/`tpub`.
fn parse(text: &str) -> Result<Xpub, String> {
    let raw = bs58::decode(text.trim())
        .into_vec()
        .map_err(|e| format!("xpub: not base58: {e}"))?;
    if raw.len() != 82 {
        return Err(format!(
            "xpub: expected 82 base58check bytes, got {}",
            raw.len()
        ));
    }
    let (payload, checksum) = raw.split_at(78);
    let digest = Sha256::digest(Sha256::digest(payload));
    if digest[..4] != *checksum {
        return Err("xpub: bad base58check checksum".to_string());
    }
    let version: [u8; 4] = payload[..4]
        .try_into()
        .map_err(|_| "xpub: internal: version slice")?;
    let version = match version {
        VERSION_XPUB => Version::Xpub,
        VERSION_TPUB => Version::Tpub,
        VERSION_XPRV | VERSION_TPRV => {
            return Err(
                "xpub: this is an extended *private* key; export the xpub instead".to_string(),
            );
        }
        other => {
            return Err(format!(
                "xpub: unknown version bytes {}",
                hex::encode(other)
            ));
        }
    };
    let depth = payload[4];
    let child = u32::from_be_bytes(
        payload[9..13]
            .try_into()
            .map_err(|_| "xpub: internal: child number slice")?,
    );
    let chain_code: [u8; 32] = payload[13..45]
        .try_into()
        .map_err(|_| "xpub: internal: chain code slice")?;
    let key = PublicKey::from_sec1_bytes(&payload[45..78])
        .map_err(|_| "xpub: key bytes are not a valid compressed secp256k1 point".to_string())?
        .to_projective();
    Ok(Xpub {
        version,
        depth,
        child,
        chain_code,
        key,
    })
}

/// The BIP352 scan public key `…/1'/0` of the account xpub `m/352'/coin'/account'/1'`,
/// with the xpub's version. Rejects an xpub at any other depth or child number.
pub fn scan_account_pubkey(xpub: &str) -> Result<(ProjectivePoint, Version), String> {
    let parsed = parse(xpub)?;
    if parsed.depth != SCAN_ACCOUNT_DEPTH || parsed.child != SCAN_ACCOUNT_CHILD {
        return Err(format!(
            "xpub: expected the node m/352'/coin'/account'/1' (depth {SCAN_ACCOUNT_DEPTH}, \
             child 1' = 0x{SCAN_ACCOUNT_CHILD:08x}), got depth {} and child 0x{:08x} ({})",
            parsed.depth,
            parsed.child,
            describe_child(parsed.child)
        ));
    }
    Ok((derive_child(&parsed, 0)?, parsed.version))
}

/// `n` or `n'` for an error message.
fn describe_child(child: u32) -> String {
    if child >= 1 << 31 {
        format!("{}'", child - (1 << 31))
    } else {
        child.to_string()
    }
}

/// Non-hardened child `index` of `parent`: `K_i = K + I_L·G` with
/// `I = HMAC-SHA512(chain_code, ser_P(K) ‖ ser32(index))`.
fn derive_child(parent: &Xpub, index: u32) -> Result<ProjectivePoint, String> {
    if index >= 1 << 31 {
        return Err(format!(
            "xpub: child {index} is hardened; only non-hardened children can be derived from an xpub"
        ));
    }
    let mut mac = Hmac::<Sha512>::new_from_slice(&parent.chain_code)
        .map_err(|_| "xpub: internal: hmac key length".to_string())?;
    mac.update(parent.key.to_affine().to_sec1_point(true).as_bytes());
    mac.update(&index.to_be_bytes());
    let i = mac.finalize().into_bytes();
    let left: [u8; 32] = i[..32]
        .try_into()
        .map_err(|_| "xpub: internal: I_L slice")?;
    let tweak = Scalar::from_repr_vartime(left.into()).ok_or_else(|| {
        "xpub: child derivation yielded I_L >= n (invalid child, try the next index)".to_string()
    })?;
    let child = parent.key + ProjectivePoint::GENERATOR * tweak;
    if bool::from(child.is_identity()) {
        return Err(
            "xpub: child derivation yielded the point at infinity (invalid child)".to_string(),
        );
    }
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::compressed;

    // BIP32 test vector 1, m/0H/1/2H and its child m/0H/1/2H/2.
    const PARENT: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    const CHILD: &str = "xpub6FHa3pjLCk84BayeJxFW2SP4XRrFd1JYnxeLeU8EqN3vDfZmbqBqaGJAyiLjTAwm6ZLRQUMv1ZACTj37sR62cfN7fe5JnJ7dh8zL4fiyLHV";
    const PARENT_PRV: &str = "xprv9z4pot5VBttmtdRTWfWQmoH1taj2axGVzFqSb8C9xaxKymcFzXBDptWmT7FwuEzG3ryjH4ktypQSAewRiNMjANTtpgP4mLTj34bhnZX7UiM";

    #[test]
    fn bip32_vector_1_child_2() {
        let parent = parse(PARENT).unwrap();
        let child = derive_child(&parent, 2).unwrap();
        assert_eq!(child, parse(CHILD).unwrap().key);
        assert_eq!(
            hex::encode(compressed(&child)),
            "02e8445082a72f29b75ca48748a914df60622a609cacfce8ed0e35804560741d29"
        );
    }

    /// The vector's parent re-serialised with other header fields.
    fn reserialize(xpub: &str, version: Option<[u8; 4]>, depth: u8, child: u32) -> String {
        let mut raw = bs58::decode(xpub).into_vec().unwrap();
        if let Some(version) = version {
            raw[..4].copy_from_slice(&version);
        }
        raw[4] = depth;
        raw[9..13].copy_from_slice(&child.to_be_bytes());
        let digest = Sha256::digest(Sha256::digest(&raw[..78]));
        raw[78..].copy_from_slice(&digest[..4]);
        bs58::encode(&raw).into_string()
    }

    #[test]
    fn parse_fields() {
        let parent = parse(PARENT).unwrap();
        assert_eq!(parent.version, Version::Xpub);
        assert_eq!(parent.depth, 3);
        assert_eq!(parent.child, 0x8000_0002);
        assert_eq!(
            hex::encode(parent.chain_code),
            "04466b9cc8e161e966409ca52986c584f07e9dc81f735db683c3ff6ec7b1503f"
        );
        assert_eq!(
            hex::encode(compressed(&parent.key)),
            "0357bfe1e341d01c69fe5654309956cbea516822fba8a601743a012a7896ee8dc2"
        );
    }

    #[test]
    fn scan_account_path_check() {
        let good = reserialize(PARENT, None, SCAN_ACCOUNT_DEPTH, SCAN_ACCOUNT_CHILD);
        let (key, version) = scan_account_pubkey(&good).unwrap();
        assert_eq!(key, derive_child(&parse(PARENT).unwrap(), 0).unwrap());
        assert_eq!(version, Version::Xpub);
        let err = scan_account_pubkey(PARENT).unwrap_err();
        assert!(err.contains("depth 3"), "{err}");
        assert!(err.contains("0x80000002 (2')"), "{err}");
        assert!(err.contains("m/352'/coin'/account'/1'"), "{err}");
        let wrong_child = reserialize(PARENT, None, SCAN_ACCOUNT_DEPTH, 1);
        let err = scan_account_pubkey(&wrong_child).unwrap_err();
        assert!(err.contains("depth 4 and child 0x00000001 (1)"), "{err}");
        let wrong_depth = reserialize(PARENT, None, 5, SCAN_ACCOUNT_CHILD);
        assert!(scan_account_pubkey(&wrong_depth).is_err());
        // derive_child itself stays path-agnostic.
        assert!(derive_child(&parse(&wrong_depth).unwrap(), 0).is_ok());
    }

    #[test]
    fn tpub_version() {
        // Re-serialise the vector's parent with the tpub version bytes.
        let tpub = reserialize(PARENT, Some(VERSION_TPUB), 3, 0x8000_0002);
        assert!(tpub.starts_with("tpub"), "{tpub}");
        let parsed = parse(&tpub).unwrap();
        assert_eq!(parsed.version, Version::Tpub);
        assert_eq!(
            derive_child(&parsed, 2).unwrap(),
            derive_child(&parse(PARENT).unwrap(), 2).unwrap()
        );
    }

    #[test]
    fn version_of_network() {
        assert_eq!(Version::of(Network::Mainnet), Version::Xpub);
        for network in [Network::Testnet, Network::Signet, Network::Regtest] {
            assert_eq!(Version::of(network), Version::Tpub);
        }
    }

    #[test]
    fn rejects_xprv_checksum_and_hardened() {
        let err = parse(PARENT_PRV).unwrap_err();
        assert!(err.contains("private"), "{err}");
        let mut broken = PARENT.to_string();
        broken.replace_range(10..11, if &PARENT[10..11] == "a" { "b" } else { "a" });
        let err = parse(&broken).unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        assert!(parse("not-base58-0OIl").is_err());
        assert!(parse("xpub").is_err());
        let err = derive_child(&parse(PARENT).unwrap(), 1 << 31).unwrap_err();
        assert!(err.contains("hardened"), "{err}");
    }
}
