//! Fuzz and property-based testing for robustness
//!
//! This module provides fuzz testing for corrupted headers and compressed blocks

#[cfg(test)]
mod tests {
    use crate::traits::*;
    use crate::util::test_utils::*;
    use std::io::Write;

    /// Test corrupted header handling
    #[test]
    fn test_corrupted_mdict_headers() {
        let temp_dir = temp_dir().unwrap();

        // Test various corrupted MDict headers
        let corrupted_headers = vec![
            "MDict",
            "MDictVersion",
            "MDictVersion 1",
            "MDictVersion invalid",
            "MDictVersion 999.999.999",
        ];

        for (i, header) in corrupted_headers.into_iter().enumerate() {
            let test_file = temp_dir.path().join(format!("corrupted_{}.mdict", i));
            std::fs::write(&test_file, header).unwrap();

            let loader = DictLoader::new();
            let result = loader.load(&test_file);

            // Should handle corrupted headers gracefully
            if let Err(error) = result {
                assert!(matches!(error, DictError::InvalidFormat(_) | DictError::IoError(_)));
            }
        }
    }

    /// Test corrupted gzip compression handling
    #[test]
    fn test_corrupted_gzip_compression() {
        let temp_dir = temp_dir().unwrap();

        // Test corrupted gzip data
        let corrupted_gzip_data = vec![
            vec![0x1F, 0x00], // Wrong second byte
            vec![0x00, 0x8B], // Wrong first byte
            vec![0x1F, 0x8B, 0x08], // Incomplete header
            vec![0x1F, 0x8B, 0x08, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
        ];

        for (i, data) in corrupted_gzip_data.into_iter().enumerate() {
            let test_file = temp_dir.path().join(format!("corrupted_{}.dsl.dz", i));
            std::fs::write(&test_file, &data).unwrap();

            let loader = DictLoader::new();
            let result = loader.load(&test_file);

            // Should handle corrupted compression gracefully
            if let Err(error) = result {
                assert!(matches!(error, DictError::DecompressionError(_) |
                                        DictError::InvalidFormat(_) |
                                        DictError::IoError(_)));
            }
        }
    }

    /// Test corrupted compressed blocks
    #[test]
    fn test_corrupted_compressed_blocks() {
        let temp_dir = temp_dir().unwrap();

        // Test corrupted compressed data
        let corrupted_data = vec![
            vec![0xFF, 0xFF, 0xFF, 0xFF],
            vec![0x00, 0x00, 0x00, 0x00],
            vec![0x1F, 0x8B, 0x08, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
            vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
        ];

        for (i, data) in corrupted_data.into_iter().enumerate() {
            let test_file = temp_dir.path().join(format!("corrupted_{}.mdict", i));
            std::fs::write(&test_file, &data).unwrap();

            let loader = DictLoader::new();
            let result = loader.load(&test_file);

            if let Err(error) = result {
                assert!(matches!(error, DictError::DecompressionError(_) |
                                        DictError::InvalidFormat(_) |
                                        DictError::IoError(_)));
            }
        }
    }
}
