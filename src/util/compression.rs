//! Compression utilities for dictionary data
//!
//! This module provides compression and decompression functions using
//! various algorithms for efficient storage and retrieval.

use std::io::{self, Read, Write, Cursor};
use std::result::Result as StdResult;

use crate::traits::{DictError, Result};

#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// GZIP compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstandard compression
    Zstd,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        CompressionAlgorithm::Zstd
    }
}

/// Compress data using the specified algorithm
pub fn compress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::Gzip => compress_gzip(data),
        CompressionAlgorithm::Lz4 => compress_lz4(data),
        CompressionAlgorithm::Zstd => compress_zstd(data),
    }
}

/// Decompress data using the specified algorithm
pub fn decompress(compressed: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(compressed.to_vec()),
        CompressionAlgorithm::Gzip => decompress_gzip(compressed),
        CompressionAlgorithm::Lz4 => decompress_lz4(compressed),
        CompressionAlgorithm::Zstd => decompress_zstd(compressed),
    }
}

/// Get compression level based on algorithm
pub fn compression_level(level: u32, algorithm: &CompressionAlgorithm) -> u32 {
    match algorithm {
        CompressionAlgorithm::Gzip => level.min(9), // GZIP levels 0-9
        CompressionAlgorithm::Zstd => level.min(19), // ZSTD levels 0-19
        _ => level,
    }
}

/// Get maximum compression level for algorithm
pub fn max_compression_level(algorithm: &CompressionAlgorithm) -> u32 {
    match algorithm {
        CompressionAlgorithm::Gzip => 9,
        CompressionAlgorithm::Zstd => 19,
        _ => 1,
    }
}

/// Get suggested compression level for size vs speed
pub fn suggested_compression_level(algorithm: &CompressionAlgorithm) -> u32 {
    match algorithm {
        CompressionAlgorithm::Gzip => 6, // Balanced default
        CompressionAlgorithm::Zstd => 6, // Balanced default
        _ => 1,
    }
}

// GZIP compression
fn compress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
    encoder.write_all(data)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    encoder.finish()
        .map_err(|e| DictError::DecompressionError(e.to_string()))
}

fn decompress_gzip(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::GzDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    Ok(decompressed)
}

// LZ4 compression
fn compress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    Ok(lz4_flex::compress(data))
}

fn decompress_lz4(compressed: &[u8]) -> Result<Vec<u8>> {
    // For decompress, we need to know the original size or use frame format
    // Since we're using block format for compression, try to decompress with estimated size
    let estimated_size = compressed.len() * 4; // Rough estimate
    lz4_flex::decompress(compressed, estimated_size)
        .map_err(|e| DictError::DecompressionError(e.to_string()))
}

// Zstandard compression
fn compress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), 6)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    encoder.write_all(data)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    encoder.finish()
        .map_err(|e| DictError::DecompressionError(e.to_string()))
}

fn decompress_zstd(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = zstd::Decoder::new(compressed)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)
        .map_err(|e| DictError::DecompressionError(e.to_string()))?;
    Ok(decompressed)
}

/// Estimate compression ratio
pub fn estimate_compression_ratio(
    original_size: u64,
    algorithm: &CompressionAlgorithm,
    level: u32,
) -> f32 {
    match algorithm {
        CompressionAlgorithm::None => 1.0,
        CompressionAlgorithm::Gzip => {
            let adjusted_level = compression_level(level, algorithm);
            // Rough estimate: GZIP typically achieves 2-3x compression
            (adjusted_level as f32 / 9.0 * 2.0 + 1.0).min(4.0)
        }
        CompressionAlgorithm::Lz4 => {
            // LZ4 is faster but less compression
            2.0 + (level as f32 / 10.0)
        }
        CompressionAlgorithm::Zstd => {
            let adjusted_level = compression_level(level, algorithm);
            // ZSTD typically achieves 3-5x compression
            (adjusted_level as f32 / 19.0 * 4.0 + 1.5).min(6.0)
        }
    }
}

/// Get algorithm-specific settings
pub fn get_algorithm_settings(algorithm: &CompressionAlgorithm) -> AlgorithmSettings {
    match algorithm {
        CompressionAlgorithm::None => AlgorithmSettings {
            supports_streaming: false,
            supports_dictionary: false,
            typical_ratio: 1.0,
            speed_category: SpeedCategory::VeryFast,
            memory_overhead: 0,
        },
        CompressionAlgorithm::Gzip => AlgorithmSettings {
            supports_streaming: true,
            supports_dictionary: false,
            typical_ratio: 2.5,
            speed_category: SpeedCategory::Fast,
            memory_overhead: 256 * 1024, // 256KB
        },
        CompressionAlgorithm::Lz4 => AlgorithmSettings {
            supports_streaming: true,
            supports_dictionary: true,
            typical_ratio: 2.0,
            speed_category: SpeedCategory::VeryFast,
            memory_overhead: 64 * 1024, // 64KB
        },
        CompressionAlgorithm::Zstd => AlgorithmSettings {
            supports_streaming: true,
            supports_dictionary: true,
            typical_ratio: 3.5,
            speed_category: SpeedCategory::Medium,
            memory_overhead: 512 * 1024, // 512KB
        },
    }
}

#[derive(Debug, Clone)]
pub struct AlgorithmSettings {
    pub supports_streaming: bool,
    pub supports_dictionary: bool,
    pub typical_ratio: f32,
    pub speed_category: SpeedCategory,
    pub memory_overhead: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpeedCategory {
    VeryFast,
    Fast,
    Medium,
    Slow,
}

/// Compress data with streaming for large datasets
pub fn compress_stream<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    algorithm: CompressionAlgorithm,
) -> Result<u64> {
    match algorithm {
        CompressionAlgorithm::None => {
            let mut buffer = vec![0u8; 8192];
            let mut total_written = 0u64;
            
            loop {
                match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        output.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            Ok(total_written)
        }
        CompressionAlgorithm::Gzip => {
            let mut encoder = flate2::write::GzEncoder::new(output, flate2::Compression::new(6));
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        encoder.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            
            encoder.finish()
                .map_err(|e| DictError::IoError(e.to_string()))?;
            Ok(total_written)
        }
        CompressionAlgorithm::Lz4 => {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(output);
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        encoder.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            
            encoder.finish()
                .map_err(|e| DictError::IoError(e.to_string()))?;
            Ok(total_written)
        }
        CompressionAlgorithm::Zstd => {
            let mut encoder = zstd::Encoder::new(output, 6)
                .map_err(|e| DictError::IoError(e.to_string()))?;
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        encoder.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            
            encoder.finish()
                .map_err(|e| DictError::IoError(e.to_string()))?;
            Ok(total_written)
        }
    }
}

/// Decompress data with streaming for large datasets
pub fn decompress_stream<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    algorithm: CompressionAlgorithm,
) -> Result<u64> {
    match algorithm {
        CompressionAlgorithm::None => {
            let mut buffer = vec![0u8; 8192];
            let mut total_written = 0u64;
            
            loop {
                match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        output.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            Ok(total_written)
        }
        CompressionAlgorithm::Gzip => {
            let mut decoder = flate2::read::GzDecoder::new(input);
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match decoder.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        output.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            Ok(total_written)
        }
        CompressionAlgorithm::Lz4 => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(input);
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match decoder.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        output.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            Ok(total_written)
        }
        CompressionAlgorithm::Zstd => {
            let mut decoder = zstd::Decoder::new(input)
                .map_err(|e| DictError::IoError(e.to_string()))?;
            let mut total_written = 0u64;
            let mut buffer = vec![0u8; 8192];
            
            loop {
                match decoder.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        output.write_all(&buffer[..n])
                            .map_err(|e| DictError::IoError(e.to_string()))?;
                        total_written += n as u64;
                    }
                    Err(e) => return Err(DictError::IoError(e.to_string())),
                }
            }
            Ok(total_written)
        }
    }
}