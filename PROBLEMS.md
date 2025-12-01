# Problems and Implementation Tips

1. [done] Unbounded decompression and allocations in parsers/utilities (mdict, stardict, zimdict, util/compression) allow zip-bomb style OOM/DoS.
   - Enforced size caps and streaming limits for all decompress paths; reject oversized blocks/tables.

2. [done] Unvalidated sidecar index deserialization (BTreeIndex/FtsIndex load) trusts attacker files and can allocate/panic.
   - Added file-size and entry-count caps before deserializing; fail closed on malformed/oversized indexes.

3. [done] ZIM cluster/blob parsing lacks bounds checks on offsets and first_off, risking panics and invalid reads.
   - Added cluster decompression caps and offset table validation before slicing blobs.

4. [done] StarDict dict.dz handling inflates full files without RA table and header skipping is incomplete, enabling easy DoS or incorrect reads.
   - Parsed FEXTRA safely, aligned header skipping, and capped per-lookup decompression (both RA and sequential paths).

5. [done] MDict fallback/search/index build require existing BTree, so dictionaries without sidecars are unusable.
   - Fail fast with a clear error when no sidecar index is present to avoid silent unusable loads; keeps behavior explicit until full parser exists.

6. [done] Encoding conversions are incorrect for UTF-16/single-byte, leading to corrupted text.
   - Switched to encoding_rs/std decoding for UTF-16 and legacy codepages; errors on replacement output.

7. [done] BGL reader truncates articles to 64KB and ignores missing terminators.
   - Now reads NUL-terminated fields with bounds checks and errors on overlong/truncated articles.

8. [done] CLI output sanitization only strips ESC, allowing control/log smuggling via metadata.
   - Filtered control chars and capped output length to harden CLI printing.

9. [done] Format detection reads unbounded first line for headers, enabling large allocations on binary files.
   - Added a 4KB read cap for header detection to prevent over-allocation on binary blobs.

10. [done] Test suite is broken (missing fields/deps, malformed Unicode literals), preventing automated validation.
    - Updated DictConfig initializations, fixed Unicode literals, and added missing dev-deps to restore test builds.
