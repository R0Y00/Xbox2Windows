// Discover and name containers, format export data, and schedule per-container work across worker threads.
// A mutex protects shared state and TSV output is sorted.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "blake3.h"
#include "sgd5.h"

namespace x2w::tool {

using Bytes = std::span<const std::uint8_t>;

// Files above this size are skipped (read whole into memory).
inline constexpr std::uint64_t MAX_FILE = std::uint64_t{2} << 30;

// X2W's `fnv_bytes` (d3d12_draw.inl): FNV-1a 64's prime, but its basis is 1469598103934665603, the
// standard basis (14695981039346656037) without its last digit. Kept as X2W computes it.
std::uint64_t fnv64(Bytes bytes);

// The container at the start of `bytes`, cut to its header's size.
Bytes bounded(Bytes bytes);

// Interpret the first 16 hexadecimal digits of the container's BLAKE3 hash as a big-endian u64.
std::uint64_t key_of(const Blake3Hash& full);

// `{:016x}`.
std::string hex16(std::uint64_t v);

// A shader container in a file: as is, or packed in an SGD5 record. Borrows the file's bytes.
struct Source {
    bool packed = false;
    // Raw: the offset of "DXBC" and the bytes from there to the end of the file.
    std::size_t offset = 0;
    Bytes data;
    // Packed.
    sgd5::Record record;

    // The container bytes; nullopt if a packed record is corrupt. `storage` holds unpacked bytes.
    std::optional<Bytes> bytes(std::vector<std::uint8_t>& storage) const;
    // "@0x1f" or "@0x1f PS_name" ("?" for a record without a name).
    std::string label() const;
};

// The packed records of SGD5 data, and every container stored as is (the executable keeps some
// records uncompressed), found by its magic and kept where dxbc::Container::parse accepts it.
std::vector<Source> sources(Bytes data);

// Visit roots depth first in directory order without following links. Include readable regular files up to
// MAX_FILE bytes, including roots that are themselves files. Path names are formatted for display.
void walk_files(const std::vector<std::filesystem::path>& roots,
                const std::function<void(const std::string& file, Bytes data)>& f);

// Format a finite float with its shortest round-trip digits. Use exponent form below 1e-4 and from 1e16
// upwards. Emit "null" for nonfinite values.
std::string json_float(float v);
// A JSON string of `s`: `"` and `\` escaped, controls as \u00XX.
std::string json_text(std::string_view s);
// String::from_utf8_lossy.
std::string utf8_lossy(Bytes bytes);
// Format a JSON array with commas between items.
template <class Items, class F>
std::string json_list(const Items& items, F f) {
    std::string o = "[";
    bool first = true;
    for (const auto& item : items) {
        if (!first) o += ",";
        first = false;
        o += f(item);
    }
    o += "]";
    return o;
}

// One container, as the export loop hands it to the translator side.
struct Found {
    std::uint64_t key = 0;
    // hex16(key).
    std::string name;
    std::uint64_t fnv = 0;
    Blake3Hash full{};
    // bounded() of the source's bytes.
    Bytes bytes;
    // The file it was first found in.
    std::string file;
};

// What the translator side made of a container.
struct Made {
    // nullopt: translated; else why not (one line of failed.tsv; tabs and newlines are replaced).
    std::optional<std::string> failed;
    std::vector<std::uint8_t> dxil;
    std::string json;
    // index.tsv's row and programs.tsv's rows for it.
    std::string index_line;
    std::vector<std::string> program_lines;
};

struct DumpOptions {
    std::filesystem::path out;
    std::vector<std::filesystem::path> paths;
    // Worker threads per file (export_x2w: default 4, at least 1).
    std::size_t jobs = 4;
    // summary.txt's `build` line.
    std::string build;
    // manifest.txt, whole, given the number of records written (none: no manifest).
    std::function<std::string(std::size_t records)> manifest;
};

struct DumpCounts {
    std::size_t containers = 0;
    std::size_t translated = 0;
    std::size_t failed = 0;
    std::size_t collisions = 0;
    std::string summary;
};

// Every container found (once by its BLAKE3) goes through `make` on the worker threads; writes <key>.dxil,
// <key>.json, index.tsv, programs.tsv, sources.tsv, failed.tsv, collisions.tsv, summary.txt and
// manifest.txt into `out`. Throws std::runtime_error when `out` cannot be made or a tsv cannot be written.
DumpCounts dump(const DumpOptions& o, const std::function<Made(const Found&)>& make);

// The header line of each tsv file dump() writes.
inline constexpr const char* INDEX_HEADER =
    "key\tstage\tcontainer_fnv64\tcontainer_blake3\tcontainer_bytes\tdxil_bytes\tcache\tthreads\tstores\t"
    "stores_by_address\tbuilder\tprograms\tfirst_source";
inline constexpr const char* PROGRAMS_HEADER = "code_fnv64\tcode_bytes\tnative_stage\twave\tform\tcontainer_key";
inline constexpr const char* SOURCES_HEADER = "key\tcontainer_fnv64\tfile\tlabel";
inline constexpr const char* FAILED_HEADER = "key\tcontainer_fnv64\tcontainer_bytes\tfirst_source\treason";
inline constexpr const char* COLLISIONS_HEADER = "key\tkept_blake3\tdropped_blake3";

}  // namespace x2w::tool
