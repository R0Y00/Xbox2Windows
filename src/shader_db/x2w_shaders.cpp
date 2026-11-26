// x2w_shaders <files or folders...> --out <cache dir> [--jobs N] [--payload <export dir>] [--max-file BYTES]
// Finds shader containers, deduplicates by key, and translates each or reads its record from the supplied export.
// Writes x2w_shaders.vxdb, manifest.txt, index.tsv, sources.tsv and misses.tsv in the cache directory.
//
// x2w_shaders --compare <export dir> <cache dir or .vxdb> [--limit N] [--list]
// Compares DXIL bytes and binding metadata. Reports the first differing disassembled function or container part,
// with counts by outcome, source module and function. Exit code 0 means all records match.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <dxcapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "archive_scan.h"
#include "startup_stubs/blake3/blake3.h"
#include "shader_db_writer.h"
#include "shader_db.h"
#include "gpu_export.h"
#include "x2w_shaders_translate.h"

#ifndef SDK_DXC_DIR
#define SDK_DXC_DIR L""
#endif

using namespace shader_db;
using Microsoft::WRL::ComPtr;

namespace {

std::wstring widen(const char* s)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring out(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), n);
    return out;
}

std::string narrow(const std::wstring& s)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string out(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, out.data(), n, nullptr, nullptr);
    return out;
}

bool write_text(const std::wstring& path, const std::string& text)
{
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    return std::fclose(f) == 0 && ok;
}

std::string hex16(uint64_t v)
{
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(v));
    return text;
}

const char* stage_name(Stage s)
{
    static const char* names[] = {"Ls", "Hs", "Es", "Gs", "Vs", "Ps", "Ms", "Cs"};
    return static_cast<uint32_t>(s) < 8 ? names[static_cast<uint32_t>(s)] : "?";
}

// ---------------------------------------------------------------------------------------------------------------
// Dump

struct Job {
    uint64_t key;
    std::vector<uint8_t> bytes;
};

int dump(const std::vector<std::wstring>& paths, const std::wstring& out_dir, unsigned jobs, const std::wstring& payload_dir,
         uint64_t max_file)
{
    const Translator* translator = payload_dir.empty() ? builtin_translator() : nullptr;
    GpuExport payload;
    if (!payload_dir.empty()) {
        if (!read_gpu_export(payload_dir, payload)) {
            std::fprintf(stderr, "x2w_shaders: cannot read the export %s\n", narrow(payload_dir).c_str());
            return 2;
        }
        std::printf("payload: %zu records from the reference export (%zu unreadable)\n", payload.rows.size(), payload.errors.size());
    } else if (!translator) {
        std::fprintf(stderr, "x2w_shaders: built without the translator (x2w_shader); pass --payload <gpu export>\n");
        return 2;
    }
    CreateDirectoryW(out_dir.c_str(), nullptr);

    std::mutex lock;
    std::condition_variable wake;
    std::deque<Job> queue;
    bool done_scanning = false;
    std::unordered_set<std::string> seen;  // deduplicate by full BLAKE3
    std::vector<ShaderRecord> records;
    std::vector<std::string> sources, misses, index;
    std::atomic<size_t> translated{0}, failed{0};
    std::string file;

    const auto finish = [&](uint64_t key, const std::vector<uint8_t>& bytes) {
        ShaderRecord r;
        std::string why;
        bool ok = false;
        if (translator) {
            ok = translator->translate(bytes.data(), bytes.size(), r, why);
        } else {
            const auto found = payload.rows.find(key);
            ok = found != payload.rows.end();
            if (ok) r = found->second.record;
            else why = "not in the payload";
        }
        r.key = key;
        r.fnv64 = fnv64(bytes.data(), bytes.size());
        r.container_bytes = static_cast<uint32_t>(bytes.size());
        std::lock_guard<std::mutex> hold(lock);
        if (!ok) {
            for (char& c : why)
                if (c == '\t' || c == '\n' || c == '\r') c = ' ';
            misses.push_back(hex16(key) + "\t" + hex16(r.fnv64) + "\t" + std::to_string(bytes.size()) + "\t" + why);
            ++failed;
            return;
        }
        index.push_back(hex16(key) + "\t" + stage_name(r.stage) + "\t" + hex16(r.fnv64) + "\t" + std::to_string(bytes.size()) + "\t" +
                        std::to_string(r.dxil.size()));
        records.push_back(std::move(r));
        ++translated;
    };

    std::vector<std::thread> workers;
    for (unsigned i = 0; i < jobs; ++i)
        workers.emplace_back([&] {
            for (;;) {
                Job job;
                {
                    std::unique_lock<std::mutex> hold(lock);
                    wake.wait(hold, [&] { return !queue.empty() || done_scanning; });
                    if (queue.empty()) return;
                    job = std::move(queue.front());
                    queue.pop_front();
                }
                wake.notify_all();
                finish(job.key, job.bytes);
            }
        });

    std::size_t containers = 0;
    scan_paths(
        paths, max_file, [&](const std::wstring& path) { file = narrow(path); },
        [&](const uint8_t* bytes, std::size_t n, const std::string& label) {
            uint8_t full[32];
            blake3_hasher h;
            blake3_hasher_init(&h);
            blake3_hasher_update(&h, bytes, n);
            blake3_hasher_finalize(&h, full, sizeof(full));
            uint64_t key = 0;
            for (int i = 0; i < 8; ++i) key = key << 8 | full[i];
            std::unique_lock<std::mutex> hold(lock);
            sources.push_back(hex16(key) + "\t" + file + "\t" + label);
            if (!seen.insert(std::string(reinterpret_cast<char*>(full), sizeof(full))).second) return;
            ++containers;
            wake.wait(hold, [&] { return queue.size() < 256; });
            queue.push_back({key, std::vector<uint8_t>(bytes, bytes + n)});
            wake.notify_all();
        });
    {
        std::lock_guard<std::mutex> hold(lock);
        done_scanning = true;
    }
    wake.notify_all();
    for (auto& w : workers) w.join();

    // Two containers with the same 64-bit key: keep the first, report the rest as misses.
    std::sort(records.begin(), records.end(), [](const ShaderRecord& a, const ShaderRecord& b) { return a.key < b.key; });
    for (std::size_t i = 1; i < records.size();)
        if (records[i].key == records[i - 1].key) {
            misses.push_back(hex16(records[i].key) + "\t" + hex16(records[i].fnv64) + "\t" + std::to_string(records[i].container_bytes) +
                             "\tkey collision");
            records.erase(records.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
            ++i;
        }

    std::string manifest = "format gpu-x2w-1\nkey blake3-16 of the D3D12_SHADER_BYTECODE bytes\n";
    if (translator) {
        manifest += std::string("producer ") + translator->name + "\n" + translator->settings();
    } else {
        manifest += "producer gpu-export\n";
        // The settings the payload's translations were made with.
        std::size_t at = 0;
        while (at < payload.manifest.size()) {
            std::size_t end = payload.manifest.find('\n', at);
            if (end == std::string::npos) end = payload.manifest.size();
            const std::string line = payload.manifest.substr(at, end - at);
            if (line.rfind("format ", 0) && line.rfind("key ", 0) && line.rfind("records ", 0) && !line.empty()) manifest += line + "\n";
            at = end + 1;
        }
    }
    manifest += "records " + std::to_string(records.size()) + "\n";

    std::string error;
    const std::size_t written = records.size();
    if (!write_db(out_dir + L"\\x2w_shaders.vxdb", std::move(records), manifest, error)) {
        std::fprintf(stderr, "x2w_shaders: %s\n", error.c_str());
        return 3;
    }
    const auto table = [](const char* header, std::vector<std::string> lines) {
        std::sort(lines.begin(), lines.end());
        std::string text = std::string(header) + "\n";
        for (const auto& l : lines) text += l + "\n";
        return text;
    };
    write_text(out_dir + L"\\manifest.txt", manifest);
    write_text(out_dir + L"\\index.tsv", table("key\tstage\tcontainer_fnv64\tcontainer_bytes\tdxil_bytes", index));
    write_text(out_dir + L"\\sources.tsv", table("key\tfile\tlabel", sources));
    write_text(out_dir + L"\\misses.tsv", table("key\tcontainer_fnv64\tcontainer_bytes\treason", misses));
    std::printf("%zu containers, %zu records written, %zu without a translation\n", containers, written, failed.load());
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Compare

struct Disassembler {
    ComPtr<IDxcCompiler3> compiler;
    Disassembler()
    {
        HMODULE dll = LoadLibraryExW(SDK_DXC_DIR L"dxcompiler.dll", nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        // X2W_DXC_DIR=<folder with dxcompiler.dll> when the build's SDK folder is not on this machine.
        if (!dll) {
            wchar_t dir[MAX_PATH];
            const DWORD n = GetEnvironmentVariableW(L"X2W_DXC_DIR", dir, MAX_PATH);
            if (n && n < MAX_PATH)
                dll = LoadLibraryExW((std::wstring(dir) + L"/dxcompiler.dll").c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        }
        if (!dll) dll = LoadLibraryW(L"dxcompiler.dll");
        auto create = dll ? reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(dll, "DxcCreateInstance")) : nullptr;
        if (create) create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    }
    bool text(const std::vector<uint8_t>& dxil, std::string& out) const
    {
        if (!compiler) return false;
        ComPtr<IDxcResult> result;
        ComPtr<IDxcBlobUtf8> blob;
        DxcBuffer input{dxil.data(), dxil.size(), 0};
        HRESULT status = E_FAIL;
        if (FAILED(compiler->Disassemble(&input, IID_PPV_ARGS(&result))) || FAILED(result->GetStatus(&status)) || FAILED(status) ||
            FAILED(result->GetOutput(DXC_OUT_DISASSEMBLY, IID_PPV_ARGS(&blob), nullptr)) || !blob)
            return false;
        out.assign(blob->GetStringPointer(), blob->GetStringLength());
        return true;
    }
};

// The disassembly cut into named pieces: "define @name" bodies, and everything else as "module".
std::vector<std::pair<std::string, std::string>> pieces(const std::string& text)
{
    std::vector<std::pair<std::string, std::string>> out;
    std::string module;
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        if (line.rfind("define ", 0) == 0) {
            const std::size_t name = line.find('@'), paren = line.find('(', name);
            std::string body = line + "\n";
            at = end + 1;
            while (at < text.size()) {
                std::size_t e = text.find('\n', at);
                if (e == std::string::npos) e = text.size();
                const std::string l = text.substr(at, e - at);
                body += l + "\n";
                at = e + 1;
                if (l == "}") break;
            }
            out.emplace_back(name == std::string::npos ? "?" : line.substr(name, paren - name), body);
            continue;
        }
        // The disassembler prints the container's hash; it differs whenever anything does.
        if (line.find("; shader hash:") == std::string::npos) module += line + "\n";
        at = end + 1;
    }
    out.emplace_back("module", module);
    return out;
}

// The first part (fourcc) of two DXBC containers that differs, or "container".
std::string first_part(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    const auto parts = [](const std::vector<uint8_t>& c) {
        std::map<std::string, std::vector<uint8_t>> out;
        if (!dxbc_size(c.data(), c.size())) return out;
        uint32_t count;
        std::memcpy(&count, c.data() + 28, 4);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t offset, size;
            std::memcpy(&offset, c.data() + 32 + 4 * i, 4);
            std::memcpy(&size, c.data() + offset + 4, 4);
            out[std::string(reinterpret_cast<const char*>(c.data() + offset), 4)].assign(c.begin() + offset + 8, c.begin() + offset + 8 + size);
        }
        return out;
    };
    const auto pa = parts(a), pb = parts(b);
    for (const auto& [name, data] : pa) {
        if (name == "HASH") continue;
        const auto other = pb.find(name);
        if (other == pb.end() || other->second != data) return "part " + name;
    }
    for (const auto& [name, data] : pb)
        if (!pa.count(name)) return "part " + name;
    return "container";
}

std::string metadata_difference(const ShaderRecord& a, const ShaderRecord& b)
{
    const auto bytes_equal = [](const auto& x, const auto& y) {
        return x.size() == y.size() && (x.empty() || !std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])));
    };
    if (a.stage != b.stage) return "stage";
    if (a.user_data_dwords != b.user_data_dwords || !bytes_equal(a.user_data, b.user_data)) return "user_data";
    if (!bytes_equal(a.samplers, b.samplers)) return "samplers";
    if (!bytes_equal(a.static_samplers, b.static_samplers)) return "static_samplers";
    if (!bytes_equal(a.root_cbvs, b.root_cbvs)) return "root_cbvs";
    if (!bytes_equal(a.root_srvs, b.root_srvs)) return "root_srvs";
    if (!bytes_equal(a.root_uavs, b.root_uavs)) return "root_uavs";
    if (!bytes_equal(a.root_indexed_views, b.root_indexed_views)) return "root_indexed_views";
    if (!bytes_equal(a.used, b.used)) return "used";
    if (a.has_threads != b.has_threads || std::memcmp(a.threads, b.threads, sizeof(a.threads))) return "threads";
    if (a.stores != b.stores || a.stores_by_address != b.stores_by_address) return "stores";
    return {};
}

std::string module_of(const std::string& path)
{
    const std::size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

int compare(const std::wstring& export_dir, std::wstring ours, std::size_t limit, bool list)
{
    GpuExport oracle;
    if (!read_gpu_export(export_dir, oracle, limit)) {
        std::fprintf(stderr, "x2w_shaders: cannot read the export %s\n", narrow(export_dir).c_str());
        return 2;
    }
    if (GetFileAttributesW(ours.c_str()) & FILE_ATTRIBUTE_DIRECTORY) ours += L"\\x2w_shaders.vxdb";
    if (!gpu_db_open(ours.c_str(), 0, 0, [](const char* line) { std::fprintf(stderr, "%s\n", line); })) return 2;
    const Disassembler dis;
    if (!dis.compiler) std::fprintf(stderr, "x2w_shaders: no dxcompiler.dll; DXIL differences are named by container part only\n");
    std::map<std::string, std::size_t> outcomes, functions;
    std::map<std::string, std::map<std::string, std::size_t>> modules;
    std::size_t mismatches = 0;
    for (const auto& [key, row] : oracle.rows) {
        const GpuShader* found = gpu_shader_by_key(key);
        std::string outcome, detail;
        if (!found) {
            outcome = "missing";
        } else {
            const ShaderRecord mine = copy_record(*found);
            if (mine.dxil != row.record.dxil) {
                outcome = "dxil differs";
                std::string a, b;
                if (dis.text(row.record.dxil, a) && dis.text(mine.dxil, b)) {
                    const auto pa = pieces(a), pb = pieces(b);
                    for (std::size_t i = 0; i < std::max(pa.size(), pb.size()) && detail.empty(); ++i)
                        if (i >= pa.size() || i >= pb.size() || pa[i] != pb[i]) detail = i < pa.size() ? pa[i].first : pb[i].first;
                    if (detail.empty()) detail = first_part(row.record.dxil, mine.dxil);
                } else {
                    detail = first_part(row.record.dxil, mine.dxil);
                }
                ++functions[detail];
            } else if (const std::string field = metadata_difference(row.record, mine); !field.empty()) {
                outcome = "metadata differs";
                detail = field;
            } else {
                outcome = "identical";
            }
        }
        ++outcomes[outcome];
        ++modules[module_of(row.first_source)][outcome];
        if (outcome != "identical") {
            ++mismatches;
            if (list) std::printf("%s\t%s\t%s\t%s\n", hex16(key).c_str(), outcome.c_str(), detail.c_str(), module_of(row.first_source).c_str());
        }
    }
    std::printf("compared %zu reference records (%zu unreadable) with %zu of ours\n", oracle.rows.size(), oracle.errors.size(), gpu_db_count());
    for (const auto& [name, n] : outcomes) std::printf("  %-18s %zu\n", name.c_str(), n);
    std::printf("by module:\n");
    for (const auto& [module, counts] : modules) {
        std::printf("  %s:", module.c_str());
        for (const auto& [name, n] : counts) std::printf(" %s %zu;", name.c_str(), n);
        std::printf("\n");
    }
    if (!functions.empty()) {
        std::printf("first differing function or part:\n");
        std::vector<std::pair<std::size_t, std::string>> sorted;
        for (const auto& [name, n] : functions) sorted.emplace_back(n, name);
        std::sort(sorted.rbegin(), sorted.rend());
        for (const auto& [n, name] : sorted) std::printf("  %6zu  %s\n", n, name.c_str());
    }
    gpu_db_close();
    return mismatches ? 1 : 0;
}

int usage()
{
    std::fprintf(stderr,
                 "usage: x2w_shaders <files or folders...> --out <cache dir> [--jobs N] [--payload <gpu export>] [--max-file BYTES]\n"
                 "       x2w_shaders --compare <gpu export> <cache dir or .vxdb> [--limit N] [--list]\n");
    return 2;
}

}  // namespace

int wmain(int argc, wchar_t** argv)
{
    std::vector<std::wstring> paths;
    std::wstring out, payload, oracle, ours;
    unsigned jobs = std::max(1u, std::thread::hardware_concurrency());
    uint64_t max_file = 2ull << 30;  // default maximum input size
    std::size_t limit = 0;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const bool more = i + 1 < argc;
        if (a == L"--out" && more) out = argv[++i];
        else if (a == L"--payload" && more) payload = argv[++i];
        else if (a == L"--jobs" && more) jobs = std::max(1, _wtoi(argv[++i]));
        else if (a == L"--max-file" && more) max_file = _wcstoui64(argv[++i], nullptr, 0);
        else if (a == L"--limit" && more) limit = static_cast<std::size_t>(_wtoi64(argv[++i]));
        else if (a == L"--list") list = true;
        else if (a == L"--compare" && i + 2 < argc) {
            oracle = argv[++i];
            ours = argv[++i];
        } else if (a.rfind(L"--", 0) == 0) return usage();
        else paths.push_back(a);
    }
    if (!oracle.empty()) return compare(oracle, ours, limit, list);
    if (paths.empty() || out.empty()) return usage();
    return dump(paths, out, jobs, payload, max_file);
}
