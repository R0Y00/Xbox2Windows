// x2w shaders <paths...> --out <dir> [--jobs N] translates each distinct container found under the paths
// (scan.h) with x2w::vs::translate. It writes <key>.dxil, <key>.json, index.tsv, programs.tsv, sources.tsv,
// failed.tsv, collisions.tsv, summary.txt and manifest.txt. Each result has cache status "translated".
// There is no orphan directory or index.
//
// Needs the translator (x2w_vs_emit) and the front end's native.h: built only with them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "scan.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/native.h"

namespace x2w::tool {

// One console program of a container: what a runtime that receives its machine code alone can hash.
struct Program {
    std::uint64_t fnv = 0;
    std::size_t bytes = 0;
    vs::native::HwStage stage{};
    bool wave64 = false;
    // "code" as stored; "patched" with the full-format export patch.
    const char* form = "code";
    vs::native::NativeProgram native;
};

std::vector<Program> programs_of(Bytes container);

// Additional pipeline creation data from the console programs.
struct Extras {
    // A compute program's thread group, when every size is known.
    std::optional<std::array<std::uint32_t, 3>> threads;
    // The descriptors the translated program reads, by root parameter.
    std::vector<vs::native::UsedResource> used;
    // Whether any program can store to buffer memory.
    bool stores = false;
    // Whether any program can store through raw addresses.
    bool stores_by_address = false;
    // A program of the ray-tracing structure builder (spilled user data).
    bool builder = false;
};

Extras extras(Bytes container);

// Whether a console program can store to buffer memory (a buffer, typed-buffer, scalar-memory, flat or
// global store or atomic, or code not all known), and whether it can store through raw addresses (flat or
// global, or code that does not decode).
bool stores(const vs::native::NativeProgram& program);
bool stores_by_address(const vs::native::NativeProgram& program);

const char* debug(vs::native::HwStage s);
std::string debug(const vs::native::RootType& r);
std::string debug(const vs::native::ResourceType& r);

// Serialize the HostShader fields, pipeline extras, and console program metadata.
std::string json(const vs::HostShader& s, const std::vector<Program>& programs, const Extras* extras,
                 const std::string& key, std::uint64_t fnv, const Blake3Hash& full, std::size_t size,
                 const char* cache);

// One container through the translator: the files and rows dump() writes for it.
Made make(const Found& found);

// Probe the D3D12 device before translation to detect WaveLaneCountMax, AMD RDNA, the null heap slot and
// raw view twins. Without a device, leave these settings unchanged. Return a log line.
std::string probe_host();

// Return translator settings as name/value pairs.
std::vector<std::pair<std::string, std::string>> settings();

// manifest.txt for `records` translations.
std::string manifest(std::size_t records);

// `x2w shaders ...` (argv after the subcommand). Returns the exit code.
int shaders_main(const std::vector<std::string>& args);

}  // namespace x2w::tool
