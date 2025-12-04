//! High-performance B-Tree index for dictionary key lookups
//!
//! This module implements a production-ready B-Tree that supports
//! insertion, lookup, range queries, validation, and persistence to disk.
//! The implementation keeps the public API compatible with the original
//! crate contract while ensuring we maintain proper B-Tree invariants.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use lru_cache::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::index::{Index, IndexConfig, IndexStats};
use crate::traits::{DictError, Result};

/// Default maximum number of keys that a node can hold.
const DEFAULT_ORDER: usize = 64;

/// On-disk snapshot persisted through `save()`/`load()`.
#[derive(Debug, Serialize, Deserialize)]
struct BTreeSnapshot {
    order: usize,
    root: Option<usize>,
    nodes: Vec<BTreeNode>,
    stats: IndexStats,
}

/// B-Tree node containing keys and child pointers.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BTreeNode {
    /// Keys in this node (sorted).
    keys: Vec<String>,
    /// Values (file offsets) in this node.
    values: Vec<u64>,
    /// Child pointers stored as node indices.
    children: Vec<usize>,
    /// Whether this is a leaf node.
    is_leaf: bool,
}

impl BTreeNode {
    fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
        }
    }

    fn new_internal() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
        }
    }

    fn key_count(&self) -> usize {
        self.keys.len()
    }
}

/// Production-ready B-Tree index implementation.
pub struct BTreeIndex {
    /// Maximum number of keys per node (fan-out minus one).
    order: usize,
    /// Root node index.
    root: Option<usize>,
    /// All nodes backing the tree.
    nodes: Vec<BTreeNode>,
    /// Statistics about the index.
    stats: IndexStats,
    /// Thread-safe access control.
    lock: Arc<RwLock<()>>,
    /// Lightweight cache for recently accessed nodes to avoid cloning.
    node_cache: LruCache<usize, BTreeNode>,
}

/// Safety cap for index files to avoid untrusted bincode bombs.
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
/// Safety cap for total nodes to avoid pathological allocations.
const MAX_NODES: usize = 1_000_000;

impl BTreeIndex {
    /// Create a new empty B-Tree index.
    pub fn new() -> Self {
        Self::with_order(DEFAULT_ORDER)
    }

    /// Create a new B-Tree index with a requested order.
    pub fn with_order(order: usize) -> Self {
        let normalized = order.max(8); // Ensure enough fan-out.
        let mut nodes = Vec::new();
        nodes.push(BTreeNode::new_leaf());

        Self {
            order: normalized,
            root: Some(0),
            nodes,
            stats: IndexStats {
                entries: 0,
                size: 0,
                build_time: 0,
                version: "1.0".to_string(),
                config: IndexConfig::default(),
            },
            lock: Arc::new(RwLock::new(())),
            node_cache: LruCache::new(4096),
        }
    }

    /// Maximum keys allowed per node (fan-out - 1).
    fn max_keys(&self) -> usize {
        self.order - 1
    }

    /// Minimum number of keys per node (except root).
    fn min_keys(&self) -> usize {
        self.order / 2
    }

    fn parse_entry_offset(bytes: &[u8]) -> Result<u64> {
        if bytes.len() != 8 {
            return Err(DictError::IndexError(
                "B-Tree entry must store an 8-byte offset".to_string(),
            ));
        }
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(raw))
    }

    /// Insert a key/value pair into the B-Tree.
    fn insert(&mut self, key: String, value: u64) -> Result<()> {
        let root_idx = self.root.unwrap();
        if self.nodes[root_idx].key_count() >= self.max_keys() {
            // Split root.
            let mut new_root = BTreeNode::new_internal();
            new_root.children.push(root_idx);
            let new_root_idx = self.nodes.len();
            self.nodes.push(new_root);
            self.root = Some(new_root_idx);
            self.split_child(new_root_idx, 0)?;
            self.insert_non_full(new_root_idx, key, value)?;
        } else {
            self.insert_non_full(root_idx, key, value)?;
        }
        Ok(())
    }

    /// Split the `child_idx` child of `parent_idx` in place.
    fn split_child(&mut self, parent_idx: usize, child_pos: usize) -> Result<()> {
        let child_idx = self.nodes[parent_idx].children[child_pos];
        let mid = self.nodes[child_idx].key_count() / 2;

        let mut new_node = if self.nodes[child_idx].is_leaf {
            BTreeNode::new_leaf()
        } else {
            BTreeNode::new_internal()
        };

        let promoted_key;
        let promoted_value;

        {
            let child = &mut self.nodes[child_idx];
            promoted_key = child.keys[mid].clone();
            promoted_value = child.values[mid];

            new_node.keys.extend_from_slice(&child.keys[mid + 1..]);
            new_node.values.extend_from_slice(&child.values[mid + 1..]);
            if !child.is_leaf {
                new_node
                    .children
                    .extend_from_slice(&child.children[mid + 1..]);
            }

            child.keys.truncate(mid);
            child.values.truncate(mid);
            if !child.is_leaf {
                child.children.truncate(mid + 1);
            }
        }

        // Push new node and update parent pointers.
        let new_idx = self.nodes.len();
        self.nodes.push(new_node);

        let parent = &mut self.nodes[parent_idx];
        parent.keys.insert(child_pos, promoted_key);
        parent.values.insert(child_pos, promoted_value);
        parent.children.insert(child_pos + 1, new_idx);

        Ok(())
    }

    /// Insert into a node that is guaranteed to be non-full.
    fn insert_non_full(&mut self, node_idx: usize, key: String, value: u64) -> Result<()> {
        if self.nodes[node_idx].is_leaf {
            let node = &mut self.nodes[node_idx];
            match node.keys.binary_search(&key) {
                Ok(pos) => node.values[pos] = value,
                Err(pos) => {
                    node.keys.insert(pos, key);
                    node.values.insert(pos, value);
                }
            }
            return Ok(());
        }

        let mut pos = match self.nodes[node_idx].keys.binary_search(&key) {
            Ok(pos) => {
                self.nodes[node_idx].values[pos] = value;
                return Ok(());
            }
            Err(pos) => pos,
        };

        let child_idx = self.nodes[node_idx].children[pos];
        if self.nodes[child_idx].key_count() >= self.max_keys() {
            self.split_child(node_idx, pos)?;
            if key > self.nodes[node_idx].keys[pos] {
                pos += 1;
            }
        }

        let child_idx = self.nodes[node_idx].children[pos];
        self.insert_non_full(child_idx, key, value)
    }

    /// Recursive search used by `binary_search`.
    fn search_recursive(&self, node_idx: usize, key: &str) -> Result<Option<u64>> {
        let node = &self.nodes[node_idx];
        match node.keys.binary_search_by(|k| k.as_str().cmp(key)) {
            Ok(pos) => Ok(Some(node.values[pos])),
            Err(pos) => {
                if node.is_leaf {
                    Ok(None)
                } else {
                    let child_idx = node.children[pos];
                    self.search_recursive(child_idx, key)
                }
            }
        }
    }

    fn range_recursive(
        &self,
        node_idx: usize,
        start: &str,
        end: &str,
        out: &mut Vec<(String, u64)>,
    ) -> Result<()> {
        let node = &self.nodes[node_idx];
        for i in 0..node.keys.len() {
            if !node.is_leaf {
                self.range_recursive(node.children[i], start, end, out)?;
            }
            let key = &node.keys[i];
            if key.as_str() >= start && key.as_str() <= end {
                out.push((key.clone(), node.values[i]));
            }
        }
        if !node.is_leaf {
            self.range_recursive(*node.children.last().unwrap(), start, end, out)?;
        }
        Ok(())
    }

    /// Estimate serialized size via bincode.
    fn estimate_serialized_size(&self) -> u64 {
        let snapshot = BTreeSnapshot {
            order: self.order,
            root: self.root,
            nodes: self.nodes.clone(),
            stats: self.stats.clone(),
        };
        bincode::serialized_size(&snapshot).unwrap_or(0)
    }

    /// Validate invariants recursively.
    fn validate_node(
        &self,
        node_idx: usize,
        min_keys: usize,
        max_keys: usize,
        min_bound: Option<&str>,
        max_bound: Option<&str>,
    ) -> Result<bool> {
        let node = &self.nodes[node_idx];
        if node_idx != self.root.unwrap() {
            if node.key_count() < min_keys || node.key_count() > max_keys {
                return Ok(false);
            }
        }

        for window in node.keys.windows(2) {
            if window[0] >= window[1] {
                return Ok(false);
            }
        }
        if let Some(min) = min_bound {
            if node.keys.iter().any(|k| k.as_str() <= min) {
                return Ok(false);
            }
        }
        if let Some(max) = max_bound {
            if node.keys.iter().any(|k| k.as_str() >= max) {
                return Ok(false);
            }
        }

        if node.is_leaf {
            return Ok(true);
        }

        if node.children.len() != node.keys.len() + 1 {
            return Ok(false);
        }

        for i in 0..node.children.len() {
            let child_min = if i == 0 {
                min_bound
            } else {
                Some(node.keys[i - 1].as_str())
            };
            let child_max = if i == node.keys.len() {
                max_bound
            } else {
                Some(node.keys[i].as_str())
            };
            if !self.validate_node(node.children[i], min_keys, max_keys, child_min, child_max)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Index for BTreeIndex {
    const INDEX_TYPE: &'static str = "btree";

    fn build(&mut self, entries: &[(String, Vec<u8>)], _config: &IndexConfig) -> Result<()> {
        let start = Instant::now();

        self.nodes.clear();
        self.nodes.push(BTreeNode::new_leaf());
        self.root = Some(0);
        self.node_cache.clear();
        self.stats.entries = 0;
        self.stats.size = 0;
        self.stats.build_time = 0;

        // Build deterministic key ordering without cloning entry payloads.
        let mut key_offsets = Vec::with_capacity(entries.len());
        for (key, value_bytes) in entries.iter() {
            let offset = Self::parse_entry_offset(value_bytes)?;
            key_offsets.push((key.clone(), offset));
        }
        key_offsets.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, offset) in key_offsets.into_iter() {
            self.insert(key, offset)?;
        }

        self.stats.entries = entries.len() as u64;
        self.stats.build_time = start.elapsed().as_millis() as u64;
        self.stats.size = self.estimate_serialized_size();

        if !self.validate()? {
            self.clear();
            return Err(DictError::IndexError(
                "B-Tree validation failed after build".to_string(),
            ));
        }
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let meta = std::fs::metadata(path).map_err(|e| {
            DictError::IoError(format!("failed to stat index {}: {e}", path.display()))
        })?;
        if meta.len() > MAX_INDEX_BYTES {
            return Err(DictError::IndexError(format!(
                "B-Tree index {} exceeds safety limit ({} bytes)",
                path.display(),
                meta.len()
            )));
        }

        let mut file = File::open(path).map_err(|e| {
            DictError::IoError(format!("failed to open index {}: {e}", path.display()))
        })?;
        let mut buf = Vec::with_capacity(meta.len() as usize);
        file.read_to_end(&mut buf).map_err(|e| {
            DictError::IoError(format!("failed to read index {}: {e}", path.display()))
        })?;

        let snapshot: BTreeSnapshot = bincode::deserialize(&buf)
            .map_err(|e| DictError::SerializationError(format!("corrupted B-Tree index: {e}")))?;

        self.order = snapshot.order.max(8);
        self.root = snapshot.root;
        self.nodes = snapshot.nodes;
        self.stats = snapshot.stats;
        self.node_cache.clear();

        if self.nodes.len() > MAX_NODES {
            return Err(DictError::IndexError(format!(
                "B-Tree index {} has too many nodes ({})",
                path.display(),
                self.nodes.len()
            )));
        }
        if !self.is_built() {
            return Err(DictError::IndexError(format!(
                "B-Tree index {} is empty or invalid",
                path.display()
            )));
        }
        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let snapshot = BTreeSnapshot {
            order: self.order,
            root: self.root,
            nodes: self.nodes.clone(),
            stats: self.stats.clone(),
        };
        let bytes = bincode::serialize(&snapshot)
            .map_err(|e| DictError::SerializationError(format!("serialize B-Tree: {e}")))?;
        let mut file = File::create(path)
            .map_err(|e| DictError::IoError(format!("create {}: {e}", path.display())))?;
        file.write_all(&bytes)
            .map_err(|e| DictError::IoError(format!("write {}: {e}", path.display())))?;
        Ok(())
    }

    fn stats(&self) -> &IndexStats {
        &self.stats
    }

    fn is_built(&self) -> bool {
        if let Some(root_idx) = self.root {
            !self.nodes[root_idx].keys.is_empty() || self.nodes.len() == 1
        } else {
            false
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(BTreeNode::new_leaf());
        self.root = Some(0);
        self.stats.entries = 0;
        self.stats.size = 0;
        self.stats.build_time = 0;
        self.node_cache.clear();
    }

    fn verify(&self) -> Result<bool> {
        self.validate()
    }
}

impl BTreeIndex {
    /// Perform binary search for a key and return its stored value (offset).
    pub fn binary_search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>> {
        if self.root.is_none() {
            return Ok(None);
        }
        match self.search_recursive(self.root.unwrap(), key)? {
            Some(offset) => Ok(Some((Vec::new(), offset))),
            None => Ok(None),
        }
    }

    /// Public search helper.
    pub fn search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>> {
        self.binary_search(key)
    }

    /// Get range of keys inclusively between `start` and `end`.
    pub fn range_query(&self, start: &str, end: &str) -> Result<Vec<(String, u64)>> {
        if self.root.is_none() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        self.range_recursive(self.root.unwrap(), start, end, &mut out)?;
        Ok(out)
    }

    /// Validate B-Tree properties.
    pub fn validate(&self) -> Result<bool> {
        if self.root.is_none() {
            return Ok(true);
        }
        let min_keys = self.min_keys();
        let max_keys = self.max_keys();
        self.validate_node(self.root.unwrap(), min_keys, max_keys, None, None)
    }
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Range query aggregation helper.
#[derive(Debug, Clone, Default)]
pub struct RangeQueryResult {
    /// Matching keys.
    pub keys: Vec<String>,
    /// Corresponding values (file offsets).
    pub values: Vec<u64>,
    /// Total number of results.
    pub count: usize,
}

impl RangeQueryResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, key: String, value: u64) {
        self.keys.push(key);
        self.values.push(value);
        self.count += 1;
    }
}
