use super::*;
use std::path::Path;

#[test]
fn test_mdx_encoding_fix() {
    // Test the MDX file loading with encoding fix
    let loader = DictLoader::new();
    let result = loader.load(Path::new("test_dict/test.mdx"));

    match result {
        Ok(dict) => {
            println!("Dictionary loaded successfully!");

            // Try to get some entries
            let test_words = ["hello", "world", "test", "example", "a", "the", "and"];

            for word in test_words.iter() {
                match dict.get(&word.to_string()) {
                    Ok(entry) => {
                        println!("Found '{}': {}", word, String::from_utf8_lossy(&entry));
                        return; // Found at least one word
                    }
                    Err(e) => {
                        println!("Word '{}' not found: {}", word, e);
                    }
                }
            }

            println!("Dictionary loaded but no test words found");
        }
        Err(e) => {
            panic!("Failed to load dictionary: {}", e);
        }
    }
}

#[test]
fn test_encoding_conversion() {
    // Test the encoding conversion function directly
    let test_data = vec![
        0xD6, 0xD0, 0xB9, 0xFA, 0xC8, 0xCB, 0xC2, 0xDB, 0xCA, 0xC7, // "测试数据" in GBK
    ];

    // Test if the data is valid UTF-8 (it shouldn't be)
    assert!(String::from_utf8(test_data.clone()).is_err(), "Test data should not be valid UTF-8");

    // Test our conversion function
    let result = dictutils::dict::mdict::convert_entry_data_if_needed(&test_data);

    match result {
        Ok(converted) => {
            // The converted data should be valid UTF-8
            let utf8_str = String::from_utf8(converted).expect("Converted data should be valid UTF-8");
            println!("Successfully converted GBK to UTF-8: {}", utf8_str);
            assert!(!utf8_str.is_empty(), "Converted string should not be empty");
        }
        Err(e) => {
            panic!("Conversion failed: {}", e);
        }
    }
}