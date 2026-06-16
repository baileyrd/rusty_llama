//! Memory-mapped checkpoint loading.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::Result;
use crate::model::Model;

/// A memory-mapped checkpoint file.
///
/// The OS pages the weights in on demand and a [`Model`] borrows directly from
/// the mapping, so there is no up-front copy of the (potentially large) weight
/// data. Keep the `Checkpoint` alive for as long as the `Model`.
pub struct Checkpoint {
    mmap: Mmap,
}

impl Checkpoint {
    /// Open and memory-map a checkpoint file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we only read the mapping, and a checkpoint is a static file
        // for the lifetime of the process. Concurrent external truncation
        // would be undefined behaviour, as with any mmap.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Checkpoint { mmap })
    }

    /// The raw mapped bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Parse the mapped bytes into a [`Model`] borrowing from this mapping.
    pub fn model(&self) -> Result<Model<'_>> {
        Model::parse(&self.mmap)
    }
}
