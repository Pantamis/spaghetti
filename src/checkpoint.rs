//! Split-key checkpoint file (`--checkpoint <PATH>`): which offset ranges a
//! split-key search has swept, so that a run killed at any moment (a spot
//! instance reclaimed, a reboot) continues where it stopped instead of
//! redoing everything or skipping anything.
//!
//! The file is small plain text, rewritten atomically (temporary file,
//! fsync, rename) every few seconds and on exit. It records the done prefix
//! (every range below is swept), the swept ranges above it (workers finish
//! ranges out of order), the tested count, the matches found so far, and the
//! search it belongs to (base key, hrp, patterns): a checkpoint is refused by
//! any other search, since its ranges were swept for its own patterns only.
//! It holds no secret (split-key mode prints tweaks, which are public).
//!
//! ```text
//! # spaghetti split-key checkpoint
//! version 1
//! base 0201e79b7d70f29abcc2c41665ac88131fe0ea7be269e558f1aac4ab78522bf51f
//! hrp sp
//! pattern sp1qqgmlnmarkets
//! layout 58 36
//! prefix 18231
//! done 18233 18236
//! tested 3757932331843584
//! found 87960930315849/1/+
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::search::{RANGE_BITS, SPLIT_RANGES};
use crate::tweak::{MAX_TWEAK_BITS, Tweak};

const VERSION: u32 = 1;

/// The search a checkpoint belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// Compressed base scan pubkey `D`.
    pub base: [u8; 33],
    pub hrp: String,
    /// Normalised pattern texts, sorted.
    pub patterns: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub identity: Identity,
    /// Every range below this one is swept.
    pub prefix: usize,
    /// Swept ranges above `prefix`, ascending.
    pub done: Vec<usize>,
    /// x candidates tested so far, earlier runs included (informational: a
    /// resumed run credits the swept ranges instead).
    pub tested: u64,
    /// Tweaks of the matches found so far, in order.
    pub found: Vec<Tweak>,
}

impl Checkpoint {
    pub fn new(identity: Identity) -> Checkpoint {
        Checkpoint {
            identity,
            prefix: 0,
            done: Vec::new(),
            tested: 0,
            found: Vec::new(),
        }
    }

    /// The checkpoint at `path`, or `None` when there is no file yet.
    pub fn load(path: &Path) -> Result<Option<Checkpoint>, String> {
        match fs::read_to_string(path) {
            Ok(text) => Checkpoint::parse(&text)
                .map(Some)
                .map_err(|e| format!("--checkpoint {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("--checkpoint {}: {e}", path.display())),
        }
    }

    /// Replaces the file at `path` atomically: a reader (or a crash) sees
    /// the old content or the new one, never a torn file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let err = |e: std::io::Error| format!("--checkpoint {}: {e}", path.display());
        let mut tmp = PathBuf::from(path);
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        tmp.set_file_name(name);
        {
            let mut file = File::create(&tmp).map_err(err)?;
            file.write_all(self.to_text().as_bytes()).map_err(err)?;
            file.sync_all().map_err(err)?;
        }
        fs::rename(&tmp, path).map_err(err)?;
        // Persist the rename itself (best effort: not every platform can
        // open a directory).
        if let Some(dir) = path.parent() {
            let dir = if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            };
            if let Ok(dir) = File::open(dir) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# spaghetti split-key checkpoint: rerun the same search with --checkpoint <this file>\n",
        );
        out.push_str(&format!("version {VERSION}\n"));
        out.push_str(&format!("base {}\n", hex::encode(self.identity.base)));
        out.push_str(&format!("hrp {}\n", self.identity.hrp));
        for pattern in &self.identity.patterns {
            out.push_str(&format!("pattern {pattern}\n"));
        }
        out.push_str(&format!("layout {MAX_TWEAK_BITS} {RANGE_BITS}\n"));
        out.push_str(&format!("prefix {}\n", self.prefix));
        // Several short lines rather than one long one.
        for chunk in self.done.chunks(32) {
            let list: Vec<String> = chunk.iter().map(usize::to_string).collect();
            out.push_str(&format!("done {}\n", list.join(" ")));
        }
        out.push_str(&format!("tested {}\n", self.tested));
        for tweak in &self.found {
            out.push_str(&format!("found {tweak}\n"));
        }
        out
    }

    pub fn parse(text: &str) -> Result<Checkpoint, String> {
        let mut version = None;
        let mut base = None;
        let mut hrp = None;
        let mut patterns = Vec::new();
        let mut layout = None;
        let mut prefix = None;
        let mut done = Vec::new();
        let mut tested = 0;
        let mut found = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let bad = |what: &str| format!("line {}: {what}: '{line}'", number + 1);
            let (key, value) = line.split_once(' ').unwrap_or((line, ""));
            let value = value.trim();
            let number = |text: &str| text.parse::<u64>().map_err(|_| bad("expected a number"));
            match key {
                "version" => version = Some(number(value)?),
                "base" => {
                    let bytes = hex::decode(value).map_err(|_| bad("expected hex"))?;
                    base = Some(
                        <[u8; 33]>::try_from(bytes.as_slice())
                            .map_err(|_| bad("expected a 33-byte key"))?,
                    );
                }
                "hrp" => hrp = Some(value.to_string()),
                "pattern" => patterns.push(value.to_string()),
                "layout" => layout = Some(value.to_string()),
                "prefix" => prefix = Some(number(value)? as usize),
                "done" => {
                    for item in value.split_whitespace() {
                        done.push(number(item)? as usize);
                    }
                }
                "tested" => tested = number(value)?,
                "found" => found.push(value.parse::<Tweak>().map_err(|e| bad(&e))?),
                _ => return Err(bad("unknown entry")),
            }
        }
        if version != Some(u64::from(VERSION)) {
            return Err(format!(
                "not a version {VERSION} spaghetti checkpoint (version {version:?})"
            ));
        }
        if layout.as_deref() != Some(format!("{MAX_TWEAK_BITS} {RANGE_BITS}").as_str()) {
            return Err(format!(
                "written for another range layout ({layout:?}; this build: {MAX_TWEAK_BITS} \
                 {RANGE_BITS}), its ranges do not apply"
            ));
        }
        let (Some(base), Some(hrp), Some(prefix)) = (base, hrp, prefix) else {
            return Err("incomplete: base, hrp and prefix are required".to_string());
        };
        if patterns.is_empty() {
            return Err("incomplete: no pattern".to_string());
        }
        if prefix > SPLIT_RANGES || done.iter().any(|&r| r <= prefix || r >= SPLIT_RANGES) {
            return Err("range numbers out of order or out of bounds".to_string());
        }
        done.sort_unstable();
        done.dedup();
        patterns.sort();
        Ok(Checkpoint {
            identity: Identity {
                base,
                hrp,
                patterns,
            },
            prefix,
            done,
            tested,
            found,
        })
    }

    /// Ranges swept in total.
    pub fn ranges_done(&self) -> usize {
        self.prefix + self.done.len()
    }
}

/// Explains why a checkpoint cannot continue this search, if it cannot.
pub fn mismatch(saved: &Identity, current: &Identity) -> Option<String> {
    if saved.base != current.base {
        return Some(format!(
            "it belongs to base key {}, not {}",
            hex::encode(saved.base),
            hex::encode(current.base)
        ));
    }
    if saved.hrp != current.hrp {
        return Some(format!(
            "it belongs to hrp {}, not {}",
            saved.hrp, current.hrp
        ));
    }
    if saved.patterns != current.patterns {
        return Some(format!(
            "it was swept for the pattern(s) {}, not {}: its ranges say nothing about these",
            saved.patterns.join(" "),
            current.patterns.join(" ")
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Checkpoint {
        Checkpoint {
            identity: Identity {
                base: [2; 33],
                hrp: "sp".to_string(),
                patterns: vec!["sp1qqgmlnmarkets".to_string(), "sp1qq?pasta".to_string()],
            },
            prefix: 18231,
            done: (18233..18300).collect(),
            tested: 3757932331843584,
            found: vec!["87960930315849/1/+".parse().unwrap()],
        }
    }

    #[test]
    fn text_round_trip() {
        let mut cp = sample();
        cp.identity.patterns.sort();
        let text = cp.to_text();
        assert!(text.contains("prefix 18231\n"), "{text}");
        assert_eq!(Checkpoint::parse(&text).unwrap(), cp);
    }

    #[test]
    fn rejects_foreign_or_broken_files() {
        let text = sample().to_text();
        for (from, to) in [
            ("version 1", "version 2"),
            ("layout 58 36", "layout 52 44"),
            ("prefix 18231", "prefix x"),
            ("done 18233", "done 17"),
            ("hrp sp\n", ""),
        ] {
            let broken = text.replacen(from, to, 1);
            assert!(Checkpoint::parse(&broken).is_err(), "{from} -> {to}");
        }
        assert!(Checkpoint::parse(&format!("{text}bogus 1\n")).is_err());
    }

    #[test]
    fn identity_mismatch_is_explained() {
        let a = sample().identity;
        let mut b = a.clone();
        assert!(mismatch(&a, &b).is_none());
        b.patterns = vec!["sp1qq?penne".to_string()];
        assert!(mismatch(&a, &b).unwrap().contains("pattern"));
        let mut c = a.clone();
        c.base[5] = 9;
        assert!(mismatch(&a, &c).unwrap().contains("base key"));
    }

    #[test]
    fn save_is_atomic_and_loads_back() {
        let dir = std::env::temp_dir().join(format!("spaghetti-cp-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("search.checkpoint");
        assert_eq!(Checkpoint::load(&path).unwrap(), None);
        let mut cp = sample();
        cp.identity.patterns.sort();
        cp.save(&path).unwrap();
        cp.prefix += 1;
        cp.done.remove(0);
        cp.save(&path).unwrap();
        assert_eq!(Checkpoint::load(&path).unwrap(), Some(cp));
        assert!(!dir.join("search.checkpoint.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
