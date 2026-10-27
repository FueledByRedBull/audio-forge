//! Identity of the pinned DeepFilterNet models.
//!
//! The C library aborts the whole process when it cannot load or parse the
//! model it is given, and that cannot be caught from Rust. A model file is
//! therefore compared with the pinned size and SHA-256 (`release-assets.json`)
//! before the library sees it. Hashing runs on the control thread that builds
//! the engine, never inside a realtime region.
//!
//! A file that was verified is not hashed again while its size and
//! modification time stay the same: engines are rebuilt for every model
//! switch and suppressor rebuild.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::SystemTime;

use crate::dsp::sha256::Sha256;

/// Expected size and lowercase hex SHA-256 of a model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ModelIdentity {
    pub size: u64,
    pub sha256: &'static str,
}

struct VerifiedFile {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
    sha256: String,
}

static VERIFIED: Mutex<Vec<VerifiedFile>> = Mutex::new(Vec::new());

/// Check `path` against `expected`; the error says what differs.
pub(super) fn verify(path: &Path, expected: &ModelIdentity) -> Result<(), String> {
    let mismatch = |detail: String| {
        format!(
            "DeepFilterNet model {} is not the pinned model ({detail})",
            path.display()
        )
    };
    let unreadable = |error: io::Error| {
        format!(
            "DeepFilterNet model {} is unreadable: {error}",
            path.display()
        )
    };
    let metadata = fs::metadata(path).map_err(unreadable)?;
    if metadata.len() != expected.size {
        return Err(mismatch(format!(
            "{} bytes, expected {}",
            metadata.len(),
            expected.size
        )));
    }
    let modified = metadata.modified().ok();
    // Held while hashing so concurrent engine builds hash a file once.
    let mut verified = VERIFIED.lock().unwrap_or_else(PoisonError::into_inner);
    let cached = verified.iter().find(|file| {
        file.path == path && file.size == metadata.len() && Some(file.modified) == modified
    });
    let sha256 = match cached {
        Some(file) => file.sha256.clone(),
        None => {
            let sha256 = hash_file(path).map_err(unreadable)?;
            if let Some(modified) = modified {
                verified.retain(|file| file.path != path);
                verified.push(VerifiedFile {
                    path: path.to_path_buf(),
                    size: metadata.len(),
                    modified,
                    sha256: sha256.clone(),
                });
            }
            sha256
        }
    };
    if sha256 != expected.sha256 {
        return Err(mismatch(format!(
            "SHA-256 {sha256}, expected {}",
            expected.sha256
        )));
    }
    Ok(())
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0_u8; 1 << 16];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            return Ok(hasher.finish_hex());
        }
        hasher.update(&chunk[..read]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::deepfilter_ffi::fake::scratch_file;
    use crate::dsp::deepfilter_ffi::DeepFilterModel;
    use crate::dsp::sha256::sha256_hex;
    use std::time::Duration;

    fn identity_of(contents: &[u8]) -> ModelIdentity {
        ModelIdentity {
            size: contents.len() as u64,
            sha256: Box::leak(sha256_hex(contents).into_boxed_str()),
        }
    }

    #[test]
    fn verify_accepts_only_the_pinned_size_and_digest() {
        let contents = b"pinned model stand-in";
        let path = scratch_file("identity-verify", contents);
        let expected = identity_of(contents);
        assert_eq!(verify(&path, &expected), Ok(()));

        let wrong_size = ModelIdentity {
            size: expected.size + 1,
            ..expected
        };
        assert!(verify(&path, &wrong_size).unwrap_err().contains("bytes"));
        let wrong_digest = ModelIdentity {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..expected
        };
        assert!(verify(&path, &wrong_digest)
            .unwrap_err()
            .contains("SHA-256"));
        assert!(verify(&path.with_file_name("missing.tar.gz"), &expected)
            .unwrap_err()
            .contains("unreadable"));
    }

    #[test]
    fn a_replaced_file_is_hashed_again() {
        let path = scratch_file("identity-replaced", b"first contents");
        let expected = identity_of(b"first contents");
        assert_eq!(verify(&path, &expected), Ok(()));

        // Same size, different bytes, later modification time.
        fs::write(&path, b"other contents").unwrap();
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        assert!(verify(&path, &expected).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn pinned_models_match_the_release_manifest_and_the_build_recipe() {
        let assets = include_str!("../../../../release-assets.json");
        let provenance = include_str!("../../../../build-support/deepfilter/provenance.json");
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let identity = model.identity();
            let entry = assets
                .find(&format!("models/{}", model.filename()))
                .map(|start| &assets[start..])
                .unwrap_or_else(|| panic!("{} missing from release-assets.json", model.filename()));
            let entry = &entry[..entry.find("\"source\"").unwrap()];
            assert!(entry.contains(&format!("\"size\": {},", identity.size)));
            assert!(entry.contains(&format!("\"sha256\": \"{}\"", identity.sha256)));
            let recipe = provenance
                .find(&format!("models/{}", model.filename()))
                .map(|start| &provenance[start..])
                .unwrap_or_else(|| panic!("{} missing from provenance.json", model.filename()));
            let recipe = &recipe[..recipe.find('}').unwrap()];
            assert!(recipe.contains(&format!("\"bytes\": {},", identity.size)));
            assert!(recipe.contains(&format!(
                "\"sha256\": \"{}\"",
                identity.sha256.to_ascii_uppercase()
            )));
        }
    }

    #[test]
    fn bundled_models_pass_their_identity_check_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let path = root.join("models").join(model.filename());
            if !path.is_file() {
                eprintln!("SKIPPED: {} is not present", path.display());
                continue;
            }
            assert_eq!(
                verify(&path, &model.identity()),
                Ok(()),
                "{}",
                path.display()
            );
        }
    }
}
