//! Encoding detection and conversion utilities
//!
//! This module provides functions for detecting text encodings,
//! converting between different encodings, and handling UTF-8 validation.

use std::collections::HashMap;
use std::result::Result as StdResult;
use std::str;

use crate::traits::{DictError, Result};

/// Supported text encodings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextEncoding {
    /// UTF-8 encoding
    Utf8,
    /// UTF-16 Little Endian
    Utf16Le,
    /// UTF-16 Big Endian
    Utf16Be,
    /// Windows-1252 (Latin-1)
    Windows1252,
    /// ISO-8859-1 (Latin-1)
    Iso88591,
    /// GBK/GB2312 (Chinese)
    Gb2312,
    /// Big5 (Traditional Chinese)
    Big5,
    /// Shift-JIS (Japanese)
    ShiftJis,
    /// EUC-KR (Korean)
    EucKr,
    /// Unknown encoding
    Unknown,
}

impl TextEncoding {
    /// Get the canonical name of the encoding
    pub fn name(&self) -> &'static str {
        match self {
            TextEncoding::Utf8 => "UTF-8",
            TextEncoding::Utf16Le => "UTF-16LE",
            TextEncoding::Utf16Be => "UTF-16BE",
            TextEncoding::Windows1252 => "Windows-1252",
            TextEncoding::Iso88591 => "ISO-8859-1",
            TextEncoding::Gb2312 => "GB2312",
            TextEncoding::Big5 => "Big5",
            TextEncoding::ShiftJis => "Shift-JIS",
            TextEncoding::EucKr => "EUC-KR",
            TextEncoding::Unknown => "Unknown",
        }
    }

    /// Check if this encoding is Unicode-based
    pub fn is_unicode(&self) -> bool {
        matches!(
            self,
            TextEncoding::Utf8 | TextEncoding::Utf16Le | TextEncoding::Utf16Be
        )
    }

    /// Check if this encoding uses variable-width characters
    pub fn is_variable_width(&self) -> bool {
        matches!(
            self,
            TextEncoding::Utf8
                | TextEncoding::Gb2312
                | TextEncoding::Big5
                | TextEncoding::ShiftJis
                | TextEncoding::EucKr
        )
    }

    /// Get the maximum byte length for a single character
    pub fn max_char_bytes(&self) -> usize {
        match self {
            TextEncoding::Utf8 => 4,
            TextEncoding::Utf16Le | TextEncoding::Utf16Be => 2,
            TextEncoding::Windows1252 | TextEncoding::Iso88591 => 1,
            TextEncoding::Gb2312
            | TextEncoding::Big5
            | TextEncoding::ShiftJis
            | TextEncoding::EucKr => 2,
            TextEncoding::Unknown => 1,
        }
    }
}

/// Detect the encoding of byte data
pub fn detect_encoding(data: &[u8]) -> Result<TextEncoding> {
    if data.is_empty() {
        return Ok(TextEncoding::Unknown);
    }

    // Check for BOM (Byte Order Mark)
    if let Some(encoding) = detect_bom(data) {
        return Ok(encoding);
    }

    // Check for UTF-8
    if is_valid_utf8(data) {
        return Ok(TextEncoding::Utf8);
    }

    // Statistical analysis for other encodings
    let mut scores = HashMap::new();

    // Score Windows-1252
    scores.insert(TextEncoding::Windows1252, score_windows1252(data));

    // Score GB2312
    scores.insert(TextEncoding::Gb2312, score_gb2312(data));

    // Score Big5
    scores.insert(TextEncoding::Big5, score_big5(data));

    // Score Shift-JIS
    scores.insert(TextEncoding::ShiftJis, score_shift_jis(data));

    // Score EUC-KR
    scores.insert(TextEncoding::EucKr, score_euc_kr(data));

    // Find encoding with highest score
    let mut best_encoding = TextEncoding::Unknown;
    let mut best_score = -1.0f32;

    for (encoding, score) in scores {
        if score > best_score {
            best_encoding = encoding;
            best_score = score;
        }
    }

    // If no encoding has a good score, fall back to UTF-8 or Windows-1252
    if best_score < 0.1 {
        if is_ascii_only(data) {
            Ok(TextEncoding::Utf8)
        } else {
            Ok(TextEncoding::Windows1252)
        }
    } else {
        Ok(best_encoding)
    }
}

/// Detect encoding using BOM (Byte Order Mark)
fn detect_bom(data: &[u8]) -> Option<TextEncoding> {
    if data.len() >= 3 && data[0..3] == [0xEF, 0xBB, 0xBF] {
        Some(TextEncoding::Utf8)
    } else if data.len() >= 2 && data[0..2] == [0xFF, 0xFE] {
        Some(TextEncoding::Utf16Le)
    } else if data.len() >= 2 && data[0..2] == [0xFE, 0xFF] {
        Some(TextEncoding::Utf16Be)
    } else {
        None
    }
}

/// Check if data is valid UTF-8
fn is_valid_utf8(data: &[u8]) -> bool {
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];

        if byte & 0x80 == 0 {
            // Single-byte character (0xxxxxxx)
            i += 1;
        } else if byte & 0xE0 == 0xC0 && i + 1 < data.len() && (data[i + 1] & 0xC0) == 0x80 {
            // Two-byte character (110xxxxx 10xxxxxx)
            i += 2;
        } else if byte & 0xF0 == 0xE0
            && i + 2 < data.len()
            && (data[i + 1] & 0xC0) == 0x80
            && (data[i + 2] & 0xC0) == 0x80
        {
            // Three-byte character (1110xxxx 10xxxxxx 10xxxxxx)
            i += 3;
        } else if byte & 0xF8 == 0xF0
            && i + 3 < data.len()
            && (data[i + 1] & 0xC0) == 0x80
            && (data[i + 2] & 0xC0) == 0x80
            && (data[i + 3] & 0xC0) == 0x80
        {
            // Four-byte character (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            i += 4;
        } else {
            // Invalid UTF-8 sequence
            return false;
        }
    }

    true
}

/// Check if data contains only ASCII characters
fn is_ascii_only(data: &[u8]) -> bool {
    data.iter().all(|&byte| byte < 0x80)
}

/// Score how well data matches Windows-1252 encoding
fn score_windows1252(data: &[u8]) -> f32 {
    let mut score = 0.0;
    let mut printable_count = 0;

    for &byte in data {
        if byte < 0x20 {
            // Control characters - may be valid in some contexts
            score += 0.1;
        } else if byte < 0x7F {
            // Printable ASCII
            printable_count += 1;
            score += 1.0;
        } else if byte >= 0xA0 && byte < 0xFF {
            // Extended ASCII characters
            score += 0.8;
        } else {
            // High ASCII that doesn't fit Windows-1252
            score -= 0.5;
        }
    }

    if printable_count > 0 {
        score / (data.len() as f32)
    } else {
        0.0
    }
}

/// Score how well data matches GB2312 encoding
fn score_gb2312(data: &[u8]) -> f32 {
    let mut score = 0.0;
    let mut i = 0;

    while i < data.len() {
        if data[i] < 0x80 {
            // ASCII character
            if data[i] >= 0x20 && data[i] != 0x7F {
                score += 1.0;
            }
            i += 1;
        } else if i + 1 < data.len() {
            // Double-byte character
            let byte1 = data[i];
            let byte2 = data[i + 1];

            if (byte1 >= 0xA1 && byte1 <= 0xF7 && byte2 >= 0xA1 && byte2 <= 0xFE)
                || (byte1 >= 0xA8 && byte1 <= 0xA8 && byte2 >= 0xA1 && byte2 <= 0xFE)
                || (byte1 >= 0xA9 && byte1 <= 0xA9 && byte2 >= 0xA1 && byte2 <= 0xFE)
            {
                score += 2.0;
            } else {
                score -= 1.0;
            }
            i += 2;
        } else {
            score -= 0.5;
            i += 1;
        }
    }

    if data.len() > 0 {
        score / (data.len() as f32 / 2.0)
    } else {
        0.0
    }
}

/// Score how well data matches Big5 encoding
fn score_big5(data: &[u8]) -> f32 {
    let mut score = 0.0;
    let mut i = 0;

    while i < data.len() {
        if data[i] < 0x80 {
            // ASCII character
            if data[i] >= 0x20 && data[i] != 0x7F {
                score += 1.0;
            }
            i += 1;
        } else if i + 1 < data.len() {
            // Double-byte character
            let byte1 = data[i];
            let byte2 = data[i + 1];

            if ((byte1 >= 0xA1 && byte1 <= 0xFE)
                && (byte2 >= 0x40 && byte2 <= 0x7E || byte2 >= 0xA1 && byte2 <= 0xFE))
                || (byte1 == 0x87 && byte2 >= 0xA1 && byte2 <= 0xFE)
            {
                score += 2.0;
            } else {
                score -= 1.0;
            }
            i += 2;
        } else {
            score -= 0.5;
            i += 1;
        }
    }

    if data.len() > 0 {
        score / (data.len() as f32 / 2.0)
    } else {
        0.0
    }
}

/// Score how well data matches Shift-JIS encoding
fn score_shift_jis(data: &[u8]) -> f32 {
    let mut score = 0.0;
    let mut i = 0;

    while i < data.len() {
        if data[i] < 0x80 {
            // Single-byte character
            if data[i] >= 0x20 && data[i] != 0x7F {
                score += 1.0;
            }
            i += 1;
        } else if data[i] >= 0x81 && data[i] <= 0x9F || data[i] >= 0xE0 && data[i] <= 0xEF {
            // First byte of double-byte character
            if i + 1 < data.len() {
                let byte2 = data[i + 1];
                if (byte2 >= 0x40 && byte2 <= 0x7E) || (byte2 >= 0x80 && byte2 <= 0xFC) {
                    score += 2.0;
                } else {
                    score -= 1.0;
                }
                i += 2;
            } else {
                score -= 0.5;
                i += 1;
            }
        } else {
            // Katakana or other single-byte characters
            score += 0.8;
            i += 1;
        }
    }

    if data.len() > 0 {
        score / (data.len() as f32 / 2.0)
    } else {
        0.0
    }
}

/// Score how well data matches EUC-KR encoding
fn score_euc_kr(data: &[u8]) -> f32 {
    let mut score = 0.0;
    let mut i = 0;

    while i < data.len() {
        if data[i] < 0x80 {
            // Single-byte character
            if data[i] >= 0x20 && data[i] != 0x7F {
                score += 1.0;
            }
            i += 1;
        } else if i + 1 < data.len() {
            // Double-byte character
            let byte1 = data[i];
            let byte2 = data[i + 1];

            if byte1 >= 0xA1 && byte1 <= 0xFE && byte2 >= 0xA1 && byte2 <= 0xFE {
                score += 2.0;
            } else {
                score -= 1.0;
            }
            i += 2;
        } else {
            score -= 0.5;
            i += 1;
        }
    }

    if data.len() > 0 {
        score / (data.len() as f32 / 2.0)
    } else {
        0.0
    }
}

/// Convert byte data from one encoding to UTF-8 string
pub fn convert_to_utf8(data: &[u8], from_encoding: TextEncoding) -> Result<String> {
    match from_encoding {
        TextEncoding::Utf8 => str::from_utf8(data)
            .map_err(|e| DictError::Internal(format!("Invalid UTF-8: {}", e)))
            .map(|s| s.to_string()),
        TextEncoding::Windows1252 => convert_windows1252_to_utf8(data),
        TextEncoding::Iso88591 => convert_iso88591_to_utf8(data),
        TextEncoding::Gb2312 => convert_gb2312_to_utf8(data),
        TextEncoding::Big5 => convert_big5_to_utf8(data),
        TextEncoding::ShiftJis => convert_shift_jis_to_utf8(data),
        TextEncoding::EucKr => convert_euc_kr_to_utf8(data),
        TextEncoding::Utf16Le => convert_utf16le_to_utf8(data),
        TextEncoding::Utf16Be => convert_utf16be_to_utf8(data),
        TextEncoding::Unknown => {
            // Try to interpret as UTF-8, fallback to Windows-1252
            if is_valid_utf8(data) {
                convert_to_utf8(data, TextEncoding::Utf8)
            } else {
                convert_windows1252_to_utf8(data)
            }
        }
    }
}

/// Convert Windows-1252 to UTF-8
fn convert_windows1252_to_utf8(data: &[u8]) -> Result<String> {
    let (cow, had_errors) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(data);
    if had_errors {
        Err(DictError::Internal(
            "Windows-1252 conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Convert ISO-8859-1 to UTF-8
fn convert_iso88591_to_utf8(data: &[u8]) -> Result<String> {
    // encoding_rs does not expose ISO-8859-1 separately; Windows-1252 is a superset and works here.
    let (cow, had_errors) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(data);
    if had_errors {
        Err(DictError::Internal(
            "ISO-8859-1 conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Convert UTF-16LE to UTF-8
fn convert_utf16le_to_utf8(data: &[u8]) -> Result<String> {
    if data.len() % 2 != 0 {
        return Err(DictError::Internal(
            "Invalid UTF-16LE data length".to_string(),
        ));
    }

    let mut u16s = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        u16s.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    String::from_utf16(&u16s)
        .map_err(|e| DictError::Internal(format!("Invalid UTF-16LE data: {e}")))
}

/// Convert UTF-16BE to UTF-8
fn convert_utf16be_to_utf8(data: &[u8]) -> Result<String> {
    if data.len() % 2 != 0 {
        return Err(DictError::Internal(
            "Invalid UTF-16BE data length".to_string(),
        ));
    }

    let mut u16s = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        u16s.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    String::from_utf16(&u16s)
        .map_err(|e| DictError::Internal(format!("Invalid UTF-16BE data: {e}")))
}

/// Simplified GB2312 to UTF-8 conversion (placeholder implementation)
fn convert_gb2312_to_utf8(_data: &[u8]) -> Result<String> {
    let (cow, _, had_errors) = encoding_rs::GBK.decode(_data);
    if had_errors {
        Err(DictError::Internal(
            "GB2312/GBK conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Simplified Big5 to UTF-8 conversion (placeholder implementation)
fn convert_big5_to_utf8(_data: &[u8]) -> Result<String> {
    let (cow, _, had_errors) = encoding_rs::BIG5.decode(_data);
    if had_errors {
        Err(DictError::Internal(
            "Big5 conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Simplified Shift-JIS to UTF-8 conversion (placeholder implementation)
fn convert_shift_jis_to_utf8(_data: &[u8]) -> Result<String> {
    let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(_data);
    if had_errors {
        Err(DictError::Internal(
            "Shift-JIS conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Simplified EUC-KR to UTF-8 conversion (placeholder implementation)
fn convert_euc_kr_to_utf8(_data: &[u8]) -> Result<String> {
    let (cow, _, had_errors) = encoding_rs::EUC_KR.decode(_data);
    if had_errors {
        Err(DictError::Internal(
            "EUC-KR conversion produced replacement characters".to_string(),
        ))
    } else {
        Ok(cow.into_owned())
    }
}

/// Validate if a string is valid UTF-8
pub fn is_valid_utf8_str(s: &str) -> bool {
    s.bytes().all(|byte| {
        byte < 0x80 || (byte & 0xE0) == 0xC0 || (byte & 0xF0) == 0xE0 || (byte & 0xF8) == 0xF0
    })
}

/// Get encoding statistics
pub fn get_encoding_stats(encoding: TextEncoding) -> EncodingStats {
    match encoding {
        TextEncoding::Utf8 => EncodingStats {
            name: encoding.name(),
            supports_unicode: true,
            max_char_size: 4,
            is_variable_width: true,
            common_in: vec!["International", "Web", "Modern files"],
        },
        TextEncoding::Windows1252 => EncodingStats {
            name: encoding.name(),
            supports_unicode: false,
            max_char_size: 1,
            is_variable_width: false,
            common_in: vec!["Windows", "Latin languages"],
        },
        TextEncoding::Iso88591 => EncodingStats {
            name: encoding.name(),
            supports_unicode: false,
            max_char_size: 1,
            is_variable_width: false,
            common_in: vec!["Unix", "Latin languages", "Old systems"],
        },
        _ => EncodingStats {
            name: encoding.name(),
            supports_unicode: false,
            max_char_size: 2,
            is_variable_width: true,
            common_in: vec!["Asian languages"],
        },
    }
}

#[derive(Debug, Clone)]
pub struct EncodingStats {
    pub name: &'static str,
    pub supports_unicode: bool,
    pub max_char_size: usize,
    pub is_variable_width: bool,
    pub common_in: Vec<&'static str>,
}
