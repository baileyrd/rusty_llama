//! Memory-mapped checkpoint loading.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::model::Model;

/// A memory-mapped checkpoint file — ordinarily one file, but a GGUF split
/// across `<prefix>-NNNNN-of-MMMMM.gguf` shards (llama.cpp's convention) maps
/// every shard and holds them all alive together.
///
/// The OS pages the weights in on demand and a [`Model`]/[`Gguf`] borrows
/// directly from the mapping(s), so there is no up-front copy of the
/// (potentially large) weight data. Keep the `Checkpoint` alive for as long
/// as anything borrowing from it.
pub struct Checkpoint {
    mmaps: Vec<Mmap>,
}

impl Checkpoint {
    fn mmap_file(path: &Path) -> Result<Mmap> {
        let file = File::open(path)?;
        // SAFETY: we only read the mapping, and a checkpoint is a static file
        // for the lifetime of the process. Concurrent external truncation
        // would be undefined behaviour, as with any mmap.
        Ok(unsafe { Mmap::map(&file)? })
    }

    /// Open and memory-map a checkpoint file. If it's a GGUF file carrying
    /// `split.count > 1` metadata, also locates and maps its sibling shards
    /// (see [`Checkpoint::gguf`]).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mmap = Self::mmap_file(path)?;

        if Gguf::is_gguf(&mmap) {
            if let Ok(head) = Gguf::parse(&mmap) {
                if let Ok(total) = head.meta_u64("split.count") {
                    if total > 1 {
                        return Self::open_sharded(path, total as u32);
                    }
                }
            }
        }
        Ok(Checkpoint { mmaps: vec![mmap] })
    }

    /// Locate and map every shard of a split GGUF, given the path to any one
    /// of them and the total shard count from its `split.count` metadata.
    fn open_sharded(path: &Path, total: u32) -> Result<Self> {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Error::Format("checkpoint path has no file name".into()))?;
        let (prefix, _this, name_total) = parse_split_name(file_name).ok_or_else(|| {
            Error::Format(format!(
                "'{file_name}' has split.count={total} metadata but its filename doesn't \
                 match the <prefix>-NNNNN-of-MMMMM.gguf convention needed to locate sibling \
                 shards"
            ))
        })?;
        if name_total != total {
            return Err(Error::Format(format!(
                "'{file_name}': filename says {name_total} total shards but split.count \
                 metadata says {total}"
            )));
        }
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let mut mmaps = Vec::with_capacity(total as usize);
        for i in 1..=total {
            let sibling = dir.join(format!("{prefix}-{i:05}-of-{total:05}.gguf"));
            mmaps.push(Self::mmap_file(&sibling).map_err(|e| {
                Error::Format(format!(
                    "loading split shard {i}/{total} ('{}'): {e}",
                    sibling.display()
                ))
            })?);
        }
        Ok(Checkpoint { mmaps })
    }

    /// The raw mapped bytes of the first (or only) shard. For a legacy
    /// llama2.c checkpoint (never sharded) or a non-split GGUF, this is the
    /// whole file; for a split GGUF it's only shard 0 — use [`Checkpoint::gguf`]
    /// for a merged view of a possibly-sharded model.
    pub fn bytes(&self) -> &[u8] {
        &self.mmaps[0]
    }

    /// Parse the mapped bytes into a [`Model`] borrowing from this mapping
    /// (legacy llama2.c checkpoints only — never sharded).
    pub fn model(&self) -> Result<Model<'_>> {
        Model::parse(&self.mmaps[0])
    }

    /// Parse into a [`Gguf`], transparently merging shards if this checkpoint
    /// is a split GGUF.
    pub fn gguf(&self) -> Result<Gguf<'_>> {
        if self.mmaps.len() == 1 {
            Gguf::parse(&self.mmaps[0])
        } else {
            let shards: Vec<&[u8]> = self.mmaps.iter().map(|m| m.as_ref()).collect();
            Gguf::parse_sharded(&shards)
        }
    }
}

/// Parse llama.cpp's split-GGUF filename convention:
/// `<prefix>-NNNNN-of-MMMMM.gguf` (5-digit, 1-indexed). Returns
/// `(prefix, this_shard, total_shards)` on a match.
fn parse_split_name(file_name: &str) -> Option<(&str, u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (head, total_str) = stem.rsplit_once("-of-")?;
    if total_str.len() != 5 || !total_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (prefix, this_str) = head.rsplit_once('-')?;
    if this_str.len() != 5 || !this_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let this_shard: u32 = this_str.parse().ok()?;
    let total: u32 = total_str.parse().ok()?;
    if this_shard == 0 || total == 0 || this_shard > total {
        return None;
    }
    Some((prefix, this_shard, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_split_name() {
        assert_eq!(
            parse_split_name("TinyLlama-1.1B-00001-of-00005.gguf"),
            Some(("TinyLlama-1.1B", 1, 5))
        );
        assert_eq!(
            parse_split_name("model-00005-of-00005.gguf"),
            Some(("model", 5, 5))
        );
    }

    #[test]
    fn rejects_non_split_names() {
        assert_eq!(parse_split_name("model.gguf"), None);
        assert_eq!(parse_split_name("model-q4_k_m.gguf"), None);
        assert_eq!(parse_split_name("model-1-of-5.gguf"), None); // not zero-padded to 5
        assert_eq!(parse_split_name("model-00000-of-00005.gguf"), None); // 1-indexed, no shard 0
        assert_eq!(parse_split_name("model-00006-of-00005.gguf"), None); // this > total
        assert_eq!(parse_split_name("model-00001-of-00005.bin"), None); // wrong extension
    }

    /// Build a minimal single-tensor GGUF file carrying `split.count` u64
    /// metadata, for exercising [`Checkpoint::open`]'s on-disk auto-discovery
    /// (as opposed to [`crate::gguf`]'s own in-memory `parse_sharded` tests).
    fn build_split_shard(tensor_name: &str, vals: &[f32], split_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        let key = "split.count";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&10u32.to_le_bytes()); // value type 10 = U64
        b.extend_from_slice(&split_count.to_le_bytes());
        b.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        b.extend_from_slice(tensor_name.as_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // type F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset (0, only one tensor per shard)
        let pad = b.len().next_multiple_of(32) - b.len();
        b.resize(b.len() + pad, 0);
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// A scratch directory unique to this test process, cleaned up on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "rusty_llama_test_{tag}_{}_{}",
                std::process::id(),
                tag.len() // cheap extra uniqueness without a random-number dependency
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn checkpoint_open_auto_discovers_split_shards() {
        let dir = TempDir::new("discover");
        let shard1 = dir.0.join("model-00001-of-00002.gguf");
        let shard2 = dir.0.join("model-00002-of-00002.gguf");
        std::fs::write(&shard1, build_split_shard("a", &[1.0, 2.0, 3.0], 2)).unwrap();
        std::fs::write(&shard2, build_split_shard("b", &[4.0, 5.0], 2)).unwrap();

        let cp = Checkpoint::open(&shard1).expect("open shard 1, discover shard 2");
        let gguf = cp.gguf().expect("merge shards");

        let a = gguf.tensor("a").expect("tensor a (shard 0)");
        let b = gguf.tensor("b").expect("tensor b (shard 1)");
        assert_eq!(a.shard, 0);
        assert_eq!(b.shard, 1);

        let want_a: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let want_b: Vec<u8> = [4.0f32, 5.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(gguf.tensor_bytes(a).unwrap(), want_a.as_slice());
        assert_eq!(gguf.tensor_bytes(b).unwrap(), want_b.as_slice());
    }

    #[test]
    fn checkpoint_open_sharded_missing_sibling_errors_cleanly() {
        let dir = TempDir::new("missing_sibling");
        let shard1 = dir.0.join("model-00001-of-00002.gguf");
        // shard2 deliberately not written.
        std::fs::write(&shard1, build_split_shard("a", &[1.0], 2)).unwrap();

        let err = match Checkpoint::open(&shard1) {
            Ok(_) => panic!("expected an error: shard 2 is missing"),
            Err(e) => format!("{e}"),
        };
        assert!(err.contains("shard 2"), "{err}");
    }

    #[test]
    fn checkpoint_open_non_split_gguf_still_works() {
        let dir = TempDir::new("non_split");
        let path = dir.0.join("model.gguf");
        std::fs::write(&path, build_split_shard("a", &[1.0, 2.0], 1)).unwrap();

        let cp = Checkpoint::open(&path).expect("open non-split file");
        let gguf = cp.gguf().expect("parse");
        assert_eq!(gguf.tensor("a").unwrap().shard, 0);
    }
}
