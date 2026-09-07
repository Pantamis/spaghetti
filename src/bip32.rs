//! Minimal BIP32: parse a serialised extended *public* key and derive one
//! non-hardened child. Enough to turn the xpub of `m/352'/coin'/account'/1'`
//! into the scan public key at `…/1'/0`.

use hmac::{Hmac, KeyInit, Mac};
use k256::elliptic_curve::Group;
use k256::elliptic_curve::PrimeField;
use k256::{ProjectivePoint, PublicKey, Scalar};
use sha2::{Digest, Sha256, Sha512};

use crate::pattern::Network;
use crate::search::compressed;

const VERSION_XPUB: [u8; 4] = [0x04, 0x88, 0xB2, 0x1E];
const VERSION_TPUB: [u8; 4] = [0x04, 0x35, 0x87, 0xCF];
const VERSION_XPRV: [u8; 4] = [0x04, 0x88, 0xAD, 0xE4];
const VERSION_TPRV: [u8; 4] = [0x04, 0x35, 0x83, 0x94];

/// The parts of a serialised extended public key that derivation needs.
#[derive(Debug)]
pub struct Xpub {
    pub network: Network,
    pub chain_code: [u8; 32],
    pub key: ProjectivePoint,
}

/// Base58check-decodes and parses an `xpub`/`tpub`.
pub fn parse(text: &str) -> Result<Xpub, String> {
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
    let network = match version {
        VERSION_XPUB => Network::Mainnet,
        VERSION_TPUB => Network::Testnet,
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
    let chain_code: [u8; 32] = payload[13..45]
        .try_into()
        .map_err(|_| "xpub: internal: chain code slice")?;
    let key = PublicKey::from_sec1_bytes(&payload[45..78])
        .map_err(|_| "xpub: key bytes are not a valid compressed secp256k1 point".to_string())?
        .to_projective();
    Ok(Xpub {
        network,
        chain_code,
        key,
    })
}

/// Non-hardened child `index` of `xpub`: `K_i = K + I_L·G` with
/// `I = HMAC-SHA512(chain_code, ser_P(K) ‖ ser32(index))`.
pub fn derive_child(xpub: &str, index: u32) -> Result<([u8; 33], Network), String> {
    if index >= 1 << 31 {
        return Err(format!(
            "xpub: child {index} is hardened; only non-hardened children can be derived from an xpub"
        ));
    }
    let parent = parse(xpub)?;
    let mut mac = Hmac::<Sha512>::new_from_slice(&parent.chain_code)
        .map_err(|_| "xpub: internal: hmac key length".to_string())?;
    mac.update(&compressed(&parent.key));
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
    Ok((compressed(&child), parent.network))
}

#[cfg(test)]
mod tests {
    use super::*;

    // BIP32 test vector 1, m/0H/1/2H and its child m/0H/1/2H/2.
    const PARENT: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    const CHILD: &str = "xpub6FHa3pjLCk84BayeJxFW2SP4XRrFd1JYnxeLeU8EqN3vDfZmbqBqaGJAyiLjTAwm6ZLRQUMv1ZACTj37sR62cfN7fe5JnJ7dh8zL4fiyLHV";
    const PARENT_PRV: &str = "xprv9z4pot5VBttmtdRTWfWQmoH1taj2axGVzFqSb8C9xaxKymcFzXBDptWmT7FwuEzG3ryjH4ktypQSAewRiNMjANTtpgP4mLTj34bhnZX7UiM";

    #[test]
    fn bip32_vector_1_child_2() {
        let (child, network) = derive_child(PARENT, 2).unwrap();
        assert_eq!(network, Network::Mainnet);
        let expected = parse(CHILD).unwrap();
        assert_eq!(child, compressed(&expected.key));
        assert_eq!(
            hex::encode(child),
            "02e8445082a72f29b75ca48748a914df60622a609cacfce8ed0e35804560741d29"
        );
    }

    #[test]
    fn parse_fields() {
        let parent = parse(PARENT).unwrap();
        assert_eq!(parent.network, Network::Mainnet);
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
    fn tpub_version() {
        // Re-serialise the vector's parent with the tpub version bytes.
        let mut raw = bs58::decode(PARENT).into_vec().unwrap();
        raw[..4].copy_from_slice(&VERSION_TPUB);
        let digest = Sha256::digest(Sha256::digest(&raw[..78]));
        raw[78..].copy_from_slice(&digest[..4]);
        let tpub = bs58::encode(&raw).into_string();
        assert!(tpub.starts_with("tpub"), "{tpub}");
        assert_eq!(parse(&tpub).unwrap().network, Network::Testnet);
        let (child, network) = derive_child(&tpub, 2).unwrap();
        assert_eq!(network, Network::Testnet);
        assert_eq!(child, derive_child(PARENT, 2).unwrap().0);
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
        let err = derive_child(PARENT, 1 << 31).unwrap_err();
        assert!(err.contains("hardened"), "{err}");
    }
}
