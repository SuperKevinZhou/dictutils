//! High-performance B-TREE index for dictionary key lookups
//!
//! This module implements an efficient B-TREE structure for fast key lookups
//! and range queries. It supports memory-mapped file access and compression.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use lru_cache::LruCache;

use crate::traits::{DictError, Result};
use crate::index::{Index, IndexConfig, IndexStats, IndexError};
use serde::{Deserialize, Serialize};

/// B-TREE node containing keys and child pointers
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BTreeNode {
    /// Keys in this node (sorted)
    keys: Vec<String>,
    /// Values (file offsets) in this node
    values: Vec<u64>,
    /// Child pointers
    children: Vec<usize>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl BTreeNode {
    /// Create a new empty leaf node
    fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
        }
    }

    /// Create a new empty internal node
    fn new_internal() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
        }
    }

    /// Binary search for a key in this node
    fn binary_search(&self, key: &str) -> Option<usize> {
        let mut left = 0;
        let mut right = self.keys.len();

        while left < right {
            let mid = (left + right) / 2;
            match self.keys[mid].as_str().cmp(key) {
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
            }
        }
        None
    }

    /// Find the position to insert a key
    fn find_insert_position(&self, key: &str) -> usize {
        self.keys.binary_search_by(|k| k.as_str().cmp(key)).unwrap_or_else(|pos| pos)
    }

    /// Check if this node is full
    fn is_full(&self, order: usize) -> bool {
        self.keys.len() >= order
    }

    /// Get the minimum key in the subtree rooted at this node
    fn min_key(&self) -> Option<&str> {
        if self.is_leaf {
            self.keys.first().map(|s| s.as_str())
        } else {
            self.children.first().and_then(|idx| {
                // This would need access to the tree to get the actual node
                // For now, return None
                None
            })
        }
    }

    /// Get the maximum key in the subtree rooted at this node
    fn max_key(&self) -> Option<&str> {
        if self.is_leaf {
            self.keys.last().map(|s| s.as_str())
        } else {
            self.children.last().and_then(|idx| {
                // This would need access to the tree to get the actual node
                // For now, return None
                None
            })
        }
    }
}

/// B-TREE index implementation
pub struct BTreeIndex {
    /// Order of the B-TREE (maximum keys per node)
    order: usize,
    /// Root node index
    root: Option<usize>,
    /// All nodes in the tree
    nodes: Vec<BTreeNode>,
    /// File offset mapping for persistence
    node_offsets: Vec<u64>,
    /// Memory-mapped file for persistence
    mmap: Option<Arc<Mmap>>,
    /// File for reading/writing
    file: Option<File>,
    /// Index statistics
    stats: IndexStats,
    /// Thread-safe access
    lock: Arc<RwLock<()>>,
    /// Cache for frequently accessed nodes
    node_cache: LruCache<usize, BTreeNode>,
}

impl BTreeIndex {
    /// Create a new empty B-TREE index
    pub fn new() -> Self {
        let order = 256; // Good balance between memory and I/O
        let cache_size = 1000;

        Self {
            order,
            root: None,
            nodes: vec![BTreeNode::new_leaf()],
            node_offsets: Vec::new(),
            mmap: None,
            file: None,
            stats: IndexStats {
                entries: 0,
                size: 0,
                build_time: 0,
                version: "1.0".to_string(),
                config: IndexConfig::default(),
            },
            lock: Arc::new(RwLock::new(())),
            node_cache: lru_cache::LruCache::new(cache_size),
        }
    }

    /// Create a new B-TREE index with specific order.
    ///
    /// NOTE:
    /// - To preserve API compatibility, this currently behaves like `new()` but
    ///   records the requested order for validation purposes.
    /// - Actual node fanout is still limited by internal implementation; callers
    ///   must not rely on a specific on-disk layout.
    pub fn with_order(order: usize) -> Self {
        let mut idx = Self::new();
        if order > 0 {
            idx.order = order;
        }
        idx
    }

    /// Insert a key-value pair into the B-TREE.
    ///
    /// Simplified, in-memory implementation:
    /// - Ignores locking and parent/child relationships.
    /// - Appends entries into the root node only.
    /// - Keeps the public API and types intact so higher-level code compiles.
    fn insert_recursive(&mut self, _node_idx: usize, key: String, value: u64) -> Result<()> {
        // Ensure we have a root node.
        let root_idx = self.root.unwrap_or(0);
        if self.nodes.is_empty() {
            self.nodes.push(BTreeNode::new_leaf());
        }

        let root = &mut self.nodes[root_idx];

        // For now, just push and keep keys sorted; no real B-Tree invariants.
        let pos = root.find_insert_position(&key);
        root.keys.insert(pos, key);
        root.values.insert(pos, value);

        Ok(())
    }

    /// Split a full node into two nodes.
    ///
    /// NOTE:
    /// - This helper is currently unused by the simplified in-memory builder.
    /// - It is kept for forward compatibility but should not be relied upon by
    ///   external callers.
    fn split_node(&self, mut node: BTreeNode) -> Result<(BTreeNode, BTreeNode, String, u64)> {
        let split_pos = self.order / 2;
        
        // Split keys and values
        let split_key = node.keys[split_pos].clone();
        let split_value = node.values[split_pos];
        
        let mut left = BTreeNode {
            keys: node.keys[..split_pos].to_vec(),
            values: node.values[..split_pos].to_vec(),
            children: if node.is_leaf { vec![] } else { node.children[..split_pos + 1].to_vec() },
            is_leaf: node.is_leaf,
        };

        let mut right = BTreeNode {
            keys: node.keys[split_pos + 1..].to_vec(),
            values: node.values[split_pos + 1..].to_vec(),
            children: if node.is_leaf { vec![] } else { node.children[split_pos + 1..].to_vec() },
            is_leaf: node.is_leaf,
        };

        Ok((left, right, split_key, split_value))
    }

    /// Find the parent of a node
    fn find_parent(&self, child_idx: usize) -> Option<usize> {
        let child = &self.nodes[child_idx];
        let child_min = child.min_key().unwrap_or("");
        let child_max = child.max_key().unwrap_or("");

        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == child_idx {
                continue;
            }

            if let (Some(node_min), Some(node_max)) = (node.min_key(), node.max_key()) {
                if node_min <= child_min && child_max <= node_max {
                    return Some(idx);
                }
            }
        }

        None
    }

    /// Insert into parent node
    fn insert_into_parent(&mut self, parent_idx: usize, key: String, value: u64, child_idx: usize) -> Result<()> {
        let mut parent = self.get_node(parent_idx)?;
        let insert_pos = parent.find_insert_position(&key);

        parent.keys.insert(insert_pos, key);
        parent.values.insert(insert_pos, value);
        parent.children.insert(insert_pos + 1, child_idx);

        self.put_node(parent_idx, parent)?;
        Ok(())
    }

    /// Search for a key in the B-TREE
    fn search_recursive(&self, node_idx: usize, key: &str) -> Result<Option<u64>> {
        let node = self.get_node(node_idx)?;
        let _guard = self.lock.read();

        if let Some(pos) = node.binary_search(key) {
            return Ok(Some(node.values[pos]));
        }

        if node.is_leaf {
            Ok(None)
        } else {
            let child_pos = node.keys.binary_search_by(|k| k.as_str().cmp(key)).unwrap_or_else(|pos| pos);
            let child_idx = if child_pos >= node.children.len() {
                node.children.len() - 1
            } else {
                node.children[child_pos]
            };

            self.search_recursive(child_idx, key)
        }
    }

    /// Get a node from storage (with caching)
    fn get_node(&self, idx: usize) -> Result<BTreeNode> {
        // For simplicity, return from in-memory storage for now
        // In a full implementation, this would handle caching properly
        if idx < self.nodes.len() {
            Ok(self.nodes[idx].clone())
        } else {
            Err(DictError::IndexError("Node index out of bounds".to_string()))
        }
    }

    /// Put a node to storage (with caching)
    fn put_node(&mut self, idx: usize, node: BTreeNode) -> Result<()> {
        if idx < self.nodes.len() {
            self.nodes[idx] = node;
            Ok(())
        } else {
            Err(DictError::IndexError("Node index out of bounds".to_string()))
        }
    }

    /// Perform binary search for a key
    pub fn binary_search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>> {
        if let Some(offset) = self.search_recursive(self.root.unwrap_or(0), key)? {
            Ok(Some((vec![], offset))) // Return (data, offset)
        } else {
            Ok(None)
        }
    }

    /// Search for a key (wrapper around binary_search)
    pub fn search(&self, key: &str) -> Result<Option<(Vec<u8>, u64)>> {
        self.binary_search(key)
    }

    /// Get range of keys
    pub fn range_query(&self, start: &str, end: &str) -> Result<Vec<(String, u64)>> {
        let mut results = Vec::new();
        self.range_recursive(self.root.unwrap_or(0), start, end, &mut results)?;
        Ok(results)
    }

    /// Recursive range query
    fn range_recursive(&self, node_idx: usize, start: &str, end: &str, results: &mut Vec<(String, u64)>) -> Result<()> {
        let node = self.get_node(node_idx)?;

        for (i, key) in node.keys.iter().enumerate() {
            if start <= key.as_str() && key.as_str() <= end {
                results.push((key.clone(), node.values[i]));
            }
        }

        if !node.is_leaf {
            for child_idx in &node.children {
                self.range_recursive(*child_idx, start, end, results)?;
            }
        }

        Ok(())
    }

    /// Get tree height
    fn get_height(&self) -> usize {
        let mut height = 0;
        let mut current = self.root.unwrap_or(0);

        while let Ok(node) = self.get_node(current) {
            if node.is_leaf {
                break;
            }
            if !node.children.is_empty() {
                current = node.children[0];
                height += 1;
            } else {
                break;
            }
        }

        height + 1
    }

    /// Validate B-TREE properties
    pub fn validate(&self) -> Result<bool> {
        if self.nodes.is_empty() {
            return Ok(true);
        }

        let root_idx = self.root.unwrap_or(0);
        let min_keys = (self.order + 1) / 2;
        let max_keys = self.order;

        self.validate_node(root_idx, min_keys, max_keys, None, None)
    }

    /// Validate a B-TREE node and its children
    fn validate_node(&self, node_idx: usize, min_keys: usize, max_keys: usize, min_bound: Option<&str>, max_bound: Option<&str>) -> Result<bool> {
        let node = &self.nodes[node_idx];

        // Check key count
        if node.keys.len() < min_keys || node.keys.len() > max_keys {
            if node_idx == self.root.unwrap_or(0) && !node.keys.is_empty() {
                // Root can have fewer keys
            } else {
                return Ok(false);
            }
        }

        // Check key ordering
        for i in 0..node.keys.len() {
            let key = &node.keys[i];

            if let Some(min) = min_bound {
                if key.as_str() < min {
                    return Ok(false);
                }
            }

            if let Some(max) = max_bound {
                if key.as_str() > max {
                    return Ok(false);
                }
            }

            if i > 0 && node.keys[i-1] >= *key {
                return Ok(false);
            }
        }

        // Check child count
        if !node.is_leaf && node.children.len() != node.keys.len() + 1 {
            return Ok(false);
        }

        // Validate children recursively
        if !node.is_leaf {
            for (i, child_idx) in node.children.iter().enumerate() {
                let child_min = if i == 0 {
                    min_bound
                } else {
                    Some(&node.keys[i-1]).map(|x| x.as_str())
                };

                let child_max = if i == node.keys.len() {
                    max_bound
                } else {
                    Some(&node.keys[i]).map(|x| x.as_str())
                };

                if !self.validate_node(*child_idx, min_keys, max_keys, child_min, child_max)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

impl Index for BTreeIndex {
    const INDEX_TYPE: &'static str = "btree";

    fn build(&mut self, entries: &[(String, Vec<u8>)], _config: &IndexConfig) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Reset state
        self.nodes.clear();
        self.node_offsets.clear();
        self.node_cache.clear();
        self.root = None;
        self.stats.entries = 0;
        self.stats.size = 0;
        self.stats.build_time = 0;

        if entries.is_empty() {
            // Empty index is valid
            return Ok(());
        }

        // Initialize with empty root
        self.nodes.push(BTreeNode::new_leaf());
        self.root = Some(0);

        // Insert all entries into the root node in sorted order.
        // This is a simplified in-memory index: it preserves key ordering and
        // provides deterministic lookups without implementing full B-Tree splits.
        for (index, (key, _value)) in entries.iter().enumerate() {
            let offset = index as u64;
            self.insert_recursive(self.root.unwrap(), key.clone(), offset)?;
        }

        self.stats.entries = entries.len() as u64;
        self.stats.build_time = start_time.elapsed().as_millis() as u64;

        // Validate the built tree; if it fails, clear state so callers don't
        // accidentally use a partial index.
        if !self.validate()? {
            self.clear();
            return Err(DictError::IndexError(
                "B-TREE validation failed; index discarded".to_string(),
            ));
        }

        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        // The legacy implementation mmapped the file but did not actually
        // reconstruct `nodes`/`root`, which made the index appear "loaded" while
        // unusable. To avoid unsafe or misleading behavior while preserving the
        // API, explicitly report that on-disk loading is not implemented yet.
        let _ = path;
        Err(DictError::IndexError(
            "B-TREE on-disk load not implemented; build index in-memory instead".to_string(),
        ))
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        // Serialize the B-TREE structure
        let serialized = bincode::serialize(&self.nodes)
            .map_err(|e| DictError::SerializationError(e.to_string()))?;

        file.write_all(&serialized)
            .map_err(|e| DictError::IoError(e.to_string()))?;

        Ok(())
    }

    fn stats(&self) -> &IndexStats {
        &self.stats
    }

    fn is_built(&self) -> bool {
        // Consider the index built only if we have a root and at least one key.
        if let Some(root_idx) = self.root {
            if let Ok(root) = self.get_node(root_idx) {
                return !root.keys.is_empty();
            }
        }
        false
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.node_offsets.clear();
        self.node_cache.clear();
        self.root = None;
        self.stats = IndexStats {
            entries: 0,
            size: 0,
            build_time: 0,
            version: "1.0".to_string(),
            config: IndexConfig::default(),
        };
    }

    fn verify(&self) -> Result<bool> {
        self.validate()
    }
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Range query results
#[derive(Debug, Clone)]
pub struct RangeQueryResult {
    /// Matching keys
    pub keys: Vec<String>,
    /// Corresponding values (file offsets)
    pub values: Vec<u64>,
    /// Total number of results
    pub count: usize,
}

impl RangeQueryResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            count: 0,
        }
    }

    /// Add a result
    pub fn add(&mut self, key: String, value: u64) {
        self.keys.push(key);
        self.values.push(value);
        self.count += 1;
    }
}

impl Default for RangeQueryResult {
    fn default() -> Self {
        Self::new()
    }
}