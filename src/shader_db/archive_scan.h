// Finds DXBC containers in standalone files, executables, DLLs and SGD5 archives. Each SGD5
// record has a preceding name, marker 0x000ddce7, packed and unpacked sizes, and a raw deflate stream. Validate
// each container's parts and take only its declared size so the bytes match the title's pipeline creation request.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace shader_db {

// `label`: "@0x<offset>" for a container as is, "@0x<offset> <record name>" for a packed record.
using ContainerSink = std::function<void(const uint8_t* bytes, std::size_t n, const std::string& label)>;

// The bytes of a well-formed DXBC container at the start of `data`, cut to its header's size; 0 if there is none.
std::size_t dxbc_size(const uint8_t* data, std::size_t available);

// Every container in one file's bytes. Returns the number of packed records that did not unpack.
std::size_t scan_bytes(const uint8_t* data, std::size_t n, const ContainerSink& sink);

// Every regular file under `paths` (files or folders, recursively) up to `max_file` bytes, mapped read-only.
// `file_sink` is told each file before its containers.
void scan_paths(const std::vector<std::wstring>& paths, uint64_t max_file, const std::function<void(const std::wstring&)>& file_sink,
                const ContainerSink& sink);

}  // namespace shader_db
