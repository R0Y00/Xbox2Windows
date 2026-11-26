// Shader export, manifest generation, and D3D12 capability probing. See shaders_dump.h.
#include "shaders_dump.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <variant>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/gfx10/mod.h"
#include "x2w_shader/spill.h"

#ifndef X2W_TOOL_SOURCE_DIR
#define X2W_TOOL_SOURCE_DIR "."
#endif

namespace x2w::tool {

namespace native = vs::native;
using native::HwStage;

namespace {

bool translated_stage(HwStage s) {
    return s == HwStage::Vs || s == HwStage::Ps || s == HwStage::Cs || s == HwStage::Ms;
}

std::string u(std::uint64_t v) { return std::to_string(v); }
const char* b(bool v) { return v ? "true" : "false"; }

// The console programs of a container's DXIL (or ACCC) part; empty when there are none or they do
// not read.
std::vector<native::NativeProgram> native_of(Bytes container) {
    try {
        const auto c = vs::dxbc::Container::parse(container);
        const auto* part = c.part(vs::dxbc::FourCC::DXIL);
        if (!part) part = c.part(vs::dxbc::FourCC::ACCC);
        if (!part) return {};
        return native::native_programs(part->data);
    } catch (const vs::ShaderError&) {
        return {};
    }
}

std::string used_json(const native::UsedResource& x) {
    return "{\"kind\":" + json_text(debug(x.kind)) + ",\"root\":" + json_text(debug(x.root)) +
           ",\"table_offset_dwords\":" + u(x.table_offset_dwords) + ",\"lower_bound\":" + u(x.lower_bound) +
           ",\"stride\":" + u(x.stride) + ",\"dwords\":" + u(x.dwords) + ",\"user_data\":" + u(x.user_data) +
           ",\"slot\":" + u(x.slot) + "}";
}

std::string threads3(const std::array<std::uint32_t, 3>& t, const char* open, const char* close) {
    return open + u(t[0]) + "," + u(t[1]) + "," + u(t[2]) + close;
}

}  // namespace

const char* debug(HwStage s) {
    switch (s) {
    case HwStage::Ls: return "Ls";
    case HwStage::Hs: return "Hs";
    case HwStage::Es: return "Es";
    case HwStage::Gs: return "Gs";
    case HwStage::Vs: return "Vs";
    case HwStage::Ps: return "Ps";
    case HwStage::Ms: return "Ms";
    case HwStage::Cs: return "Cs";
    }
    return "?";
}

std::string debug(const native::RootType& r) {
    using K = native::RootType::Kind;
    switch (r.kind) {
    case K::Constants: return "Constants";
    case K::Descriptor: return "Descriptor";
    case K::Table: return "Table";
    case K::ImmediateSampler: return "ImmediateSampler";
    case K::ResourceHeap: return "ResourceHeap";
    case K::SamplerHeap: return "SamplerHeap";
    case K::Other: return "Other(" + u(r.other) + ")";
    }
    return "?";
}

std::string debug(const native::ResourceType& r) {
    using K = native::ResourceType::Kind;
    switch (r.kind) {
    case K::Srv: return "Srv";
    case K::Sampler: return "Sampler";
    case K::Cbv: return "Cbv";
    case K::Uav: return "Uav";
    case K::Other: return "Other(" + u(r.other) + ")";
    }
    return "?";
}

std::vector<Program> programs_of(Bytes container) {
    std::vector<Program> out;
    for (auto& p : native_of(container)) {
        const std::vector<std::uint8_t> patched = p.patched_code();
        if (patched != p.code) {
            Program q;
            q.fnv = fnv64(patched);
            q.bytes = patched.size();
            q.stage = p.stage;
            q.wave64 = p.wave64;
            q.form = "patched";
            q.native = p;
            out.push_back(std::move(q));
        }
        Program q;
        q.fnv = fnv64(p.code);
        q.bytes = p.code.size();
        q.stage = p.stage;
        q.wave64 = p.wave64;
        q.form = "code";
        q.native = std::move(p);
        out.push_back(std::move(q));
    }
    return out;
}

bool stores(const native::NativeProgram& program) {
    using vs::gfx10::Encoding;
    std::vector<vs::gfx10::Instruction> code;
    try {
        code = vs::gfx10::decode(program.code);
    } catch (const vs::ShaderError&) {
        return true;
    }
    for (const auto& i : code) {
        const auto name = i.name();
        switch (i.encoding) {
        case Encoding::Mubuf:
        case Encoding::Mtbuf:
        case Encoding::Smem:
        case Encoding::Flat:
        case Encoding::Global:
            if (!name || ((name->find("store") != std::string_view::npos ||
                           name->find("atomic") != std::string_view::npos) &&
                          name->find("scratch") == std::string_view::npos))
                return true;
            break;
        case Encoding::Sop1:
            if (!name || *name == "s_setpc_b64" || *name == "s_swappc_b64") return true;
            break;
        default: break;
        }
    }
    return false;
}

bool stores_by_address(const native::NativeProgram& program) {
    using vs::gfx10::Encoding;
    std::vector<vs::gfx10::Instruction> code;
    try {
        code = vs::gfx10::decode(program.code);
    } catch (const vs::ShaderError&) {
        return true;
    }
    for (const auto& i : code) {
        if (i.encoding != Encoding::Flat && i.encoding != Encoding::Global) continue;
        const auto name = i.name();
        if (!name || ((name->find("store") != std::string_view::npos ||
                       name->find("atomic") != std::string_view::npos) &&
                      name->find("scratch") == std::string_view::npos))
            return true;
    }
    return false;
}

Extras extras(Bytes container) {
    const std::vector<native::NativeProgram> programs = native_of(container);
    Extras e;
    for (const auto& p : programs) {
        if (const auto* cs = std::get_if<native::ComputeIo>(&p.io)) {
            if (cs->threads[0] > 0 && cs->threads[1] > 0 && cs->threads[2] > 0) {
                e.threads = cs->threads;
                break;
            }
        }
    }
    for (const auto& p : programs) {
        if (translated_stage(p.stage)) {
            e.used = p.used;
            break;
        }
    }
    e.stores = programs.empty();
    e.stores_by_address = programs.empty();
    for (const auto& p : programs) e.stores = e.stores || stores(p);
    for (const auto& p : programs) e.stores_by_address = e.stores_by_address || stores_by_address(p);
    for (const auto& p : programs) e.builder = e.builder || vs::spill::is_spilled(p);
    return e;
}

std::string json(const vs::HostShader& s, const std::vector<Program>& programs, const Extras* extras,
                 const std::string& key, std::uint64_t fnv, const Blake3Hash& full, std::size_t size,
                 const char* cache) {
    std::string o = "{\n";
    o += "  \"key\": " + json_text(key) + ",\n";
    o += "  \"container_fnv64\": \"" + hex16(fnv) + "\",\n";
    o += "  \"container_blake3\": " + json_text(to_hex(full)) + ",\n";
    o += "  \"container_bytes\": " + u(size) + ",\n";
    o += "  \"cache\": " + json_text(cache) + ",\n";
    o += "  \"stage\": " + json_text(debug(s.stage)) + ",\n";
    o += "  \"dxil_bytes\": " + u(s.dxil.size()) + ",\n";
    o += "  \"user_data_dwords\": " + u(s.user_data_dwords()) + ",\n";
    const std::string ud = json_list(s.user_data, [](const vs::UserData& x) {
        return "{\"index\":" + u(x.index) + ",\"slot\":" + u(x.slot) + ",\"root\":" + json_text(debug(x.root)) +
               ",\"dword\":" + u(x.dword) + "}";
    });
    o += "  \"user_data\": " + ud + ",\n";
    o += "  \"samplers\": " +
         json_list(s.samplers, [](const std::array<std::uint32_t, 4>& w) {
             return json_list(w, [](std::uint32_t v) { return u(v); });
         }) +
         ",\n";
    const std::string ss = json_list(s.static_samplers, [](const vs::StaticSampler& t) {
        return "{\"filter\":" + u(t.filter) + ",\"address\":[" + u(t.address[0]) + "," + u(t.address[1]) + "," +
               u(t.address[2]) + "],\"mip_lod_bias\":" + json_float(t.mip_lod_bias) +
               ",\"max_anisotropy\":" + u(t.max_anisotropy) + ",\"comparison\":" + u(t.comparison) +
               ",\"border\":" + u(t.border) + ",\"min_lod\":" + json_float(t.min_lod) +
               ",\"max_lod\":" + json_float(t.max_lod) + "}";
    });
    o += "  \"static_samplers\": " + ss + ",\n";
    auto bytes = [](const std::vector<std::uint8_t>& v) { return json_list(v, [](std::uint8_t x) { return u(x); }); };
    o += "  \"root_cbvs\": " + bytes(s.root_cbvs) + ",\n";
    o += "  \"root_srvs\": " + bytes(s.root_srvs) + ",\n";
    o += "  \"root_uavs\": " + bytes(s.root_uavs) + ",\n";
    const std::string riv = json_list(s.root_indexed_views, [](const vs::resource::RootIndexedView& v) {
        return "{\"table_slot\":" + u(v.table_slot) + ",\"constant_slot\":" + u(v.constant_slot) +
               ",\"constant_dword\":" + u(v.constant_dword) + ",\"byte_offset\":" + u(v.byte_offset) +
               ",\"shift\":" + u(v.shift) + "}";
    });
    o += "  \"root_indexed_views\": " + riv + ",\n";
    // Pipeline creation's own reading of the programs (`pipeline::translate`).
    if (extras) {
        const std::string threads = extras->threads ? threads3(*extras->threads, "[", "]") : "null";
        o += "  \"threads\": " + threads + ",\n";
        o += "  \"used\": " + json_list(extras->used, used_json) + ",\n";
        o += std::string("  \"stores\": ") + b(extras->stores) + ",\n";
        o += std::string("  \"stores_by_address\": ") + b(extras->stores_by_address) + ",\n";
        o += std::string("  \"builder\": ") + b(extras->builder) + ",\n";
    }
    // Derived, not HostShader fields: the emitted container's parts, and the console program the
    // translation is made from.
    std::string parts = "[]";
    try {
        const auto c = vs::dxbc::Container::parse(s.dxil);
        parts = json_list(c.parts, [](const vs::dxbc::Part& p) {
            return "{\"fourcc\":" + json_text(utf8_lossy(Bytes(p.fourcc.b))) + ",\"bytes\":" + u(p.data.size()) +
                   "}";
        });
    } catch (const vs::ShaderError&) {
    }
    o += "  \"derived_dxil_parts\": " + parts + ",\n";
    const Program* translated = nullptr;
    for (const auto& p : programs) {
        if (std::string_view(p.form) == "code" && translated_stage(p.stage)) {
            translated = &p;
            break;
        }
    }
    if (translated) {
        const native::NativeProgram& n = translated->native;
        std::string threads = "null";
        if (const auto* cs = std::get_if<native::ComputeIo>(&n.io)) threads = threads3(cs->threads, "[", "]");
        else if (const auto* ms = std::get_if<native::MeshIo>(&n.io)) threads = "[" + u(ms->threads) + ",1,1]";
        const std::string layout = json_list(n.layout, [](const native::RootParam& r) {
            return "{\"root\":" + json_text(debug(r.root)) + ",\"table_offset_dwords\":" + u(r.table_offset_dwords) +
                   ",\"sgprs\":" + u(r.sgprs) + ",\"base_register\":" + u(r.base_register) +
                   ",\"user_data\":" + u(r.user_data) + ",\"slot\":" + u(r.slot) + "}";
        });
        const std::string used = json_list(n.used, used_json);
        o += "  \"derived_native\": {\"stage\":" + json_text(debug(n.stage)) +
             ",\"wave_size\":" + (n.wave64 ? "64" : "32") + ",\"vgprs\":" + u(n.vgprs) + ",\"threads\":" + threads +
             ",\"code_bytes\":" + u(translated->bytes) + ",\"code_fnv64\":\"" + hex16(translated->fnv) +
             "\",\"layout\":" + layout + ",\"used\":" + used + "},\n";
    } else {
        o += "  \"derived_native\": null,\n";
    }
    const std::string progs = json_list(programs, [](const Program& p) {
        return "{\"stage\":" + json_text(debug(p.stage)) + ",\"form\":" + json_text(p.form) +
               ",\"wave_size\":" + (p.wave64 ? "64" : "32") + ",\"code_bytes\":" + u(p.bytes) +
               ",\"code_fnv64\":\"" + hex16(p.fnv) + "\"}";
    });
    o += "  \"derived_programs\": " + progs + "\n";
    o += "}\n";
    return o;
}

Made make(const Found& f) {
    Made m;
    const std::vector<Program> programs = programs_of(f.bytes);
    vs::HostShader shader;
    try {
        shader = vs::translate(f.bytes);
    } catch (const vs::ShaderError& e) {
        m.failed = e.to_string();
        return m;
    }
    const char* cache_word = "translated";
    const Extras ex = extras(f.bytes);
    m.json = json(shader, programs, &ex, f.name, f.fnv, f.full, f.bytes.size(), cache_word);
    const std::string threads = ex.threads ? threads3(*ex.threads, "", "") : "-";
    m.index_line = f.name + "\t" + debug(shader.stage) + "\t" + hex16(f.fnv) + "\t" + to_hex(f.full) + "\t" +
                   u(f.bytes.size()) + "\t" + u(shader.dxil.size()) + "\t" + cache_word + "\t" + threads + "\t" +
                   (ex.stores ? "1" : "0") + "\t" + (ex.stores_by_address ? "1" : "0") + "\t" +
                   (ex.builder ? "1" : "0") + "\t" + u(programs.size()) + "\t" + f.file;
    for (const auto& p : programs)
        m.program_lines.push_back(hex16(p.fnv) + "\t" + u(p.bytes) + "\t" + debug(p.stage) + "\t" +
                                  (p.wave64 ? "64" : "32") + "\t" + p.form + "\t" + f.name);
    m.dxil = std::move(shader.dxil);
    return m;
}

namespace {

using Microsoft::WRL::ComPtr;

constexpr std::uint32_t PORTABLE_VIEWS = 1'000'000;
constexpr std::uint32_t UNCAPPED_VIEWS = 1u << 21;
constexpr std::uint32_t RAW_TWINS = 1u << 21;

std::optional<std::string> env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
}

}  // namespace

std::string probe_host() {
    // Create the default adapter device at feature level 12_0.
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
        return "no host D3D12 device: the translator keeps its defaults";
    std::string log;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof o1))) {
        vs::lower::set_host_max_wave(o1.WaveLaneCountMax);
        // Identify the device adapter by its LUID.
        bool amd = false;
        ComPtr<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
            ComPtr<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(adapter->GetDesc1(&desc)))
                amd = desc.VendorId == 0x1002;
        }
        // (RDNA has waves of 32 and 64, the vendor's older cards of 64 alone.)
        vs::lower::set_host_rdna(amd && o1.WaveLaneCountMin == 32 && o1.WaveLaneCountMax == 64);
        log += "host wave lanes " + u(o1.WaveLaneCountMin) + ".." + u(o1.WaveLaneCountMax) + (amd ? ", AMD" : "");
    }
    // Probe the supported view heap limit.
    auto fits = [&](std::uint32_t count) {
        D3D12_DESCRIPTOR_HEAP_DESC d{};
        d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        d.NumDescriptors = count;
        d.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> heap;
        return SUCCEEDED(device->CreateDescriptorHeap(&d, IID_PPV_ARGS(&heap)));
    };
    const auto raw = env("X2W_SHADERCFG_RAW_VIEWS");
    const bool raw_twins = !raw || *raw != "0";
    const bool twins = raw_twins && fits(2 * RAW_TWINS);
    if (twins) vs::shader::set_raw_views(RAW_TWINS);
    // (The twins lie past the null slot.)
    const std::uint32_t views = twins ? RAW_TWINS : fits(UNCAPPED_VIEWS) ? UNCAPPED_VIEWS : PORTABLE_VIEWS;
    vs::shader::set_view_heap_limit(views - 1);
    log += "; host view heaps hold " + u(views) + " views" + (twins ? ", raw twins" : "");
    return log;
}

std::vector<std::pair<std::string, std::string>> settings() {
    const auto host = vs::lower::host_settings();
    std::string translator;
    for (std::uint8_t x : vs::translation_fingerprint()) {
        char buf[3];
        std::snprintf(buf, sizeof buf, "%02x", x);
        translator += buf;
    }
    std::vector<std::pair<std::string, std::string>> out = {
        {"translator", translator},
        {"max_wave", u(host[0])},
        {"view_heap_limit", u(host[1])},
        {"raw_views", u(host[2])},
        {"host_rdna", u(host[3])},
        {"loop_limit", u(vs::loop_limit())},
        {"group_sync", u(vs::group_sync())},
        {"shader_opt", vs::hex_alt(vs::shader_opt())},
    };
    for (const char* name :
         {"X2W_SHADERCFG_WAVE_WIDE", "X2W_SHADERCFG_WAVE_DEEP_SKIPS", "X2W_SHADERCFG_WAVE_OWN_BITS", "X2W_SHADERCFG_WAVE_LOCAL",
          "X2W_SHADERCFG_F16_NEAREST", "X2W_SHADERCFG_HELPER_OWN", "X2W_SHADERCFG_FIRSTLANE_OWN", "X2W_SHADERCFG_FIRST_HALF",
          "X2W_SHADERCFG_STRICT_JOINS", "X2W_SHADERCFG_MESH_OUTPUT_STRICT", "X2W_SHADERCFG_BVH_FENCES", "X2W_SHADERCFG_BVH_BOUNDS"})
        out.emplace_back(name, env(name).value_or(""));
    return out;
}

namespace {

std::string build_hex() {
    const auto fp = vs::translation_fingerprint();
    return to_hex(blake3(fp));
}

// Record the source tree commit and whether the working tree has changes.
std::string commit() {
    auto git = [](const std::string& args) {
        const std::string cmd = std::string("git -C \"") + X2W_TOOL_SOURCE_DIR + "\" " + args + " 2>NUL";
        std::string out;
        if (FILE* p = _popen(cmd.c_str(), "r")) {
            char buf[256];
            while (std::fgets(buf, sizeof buf, p)) out += buf;
            _pclose(p);
        }
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
        return out;
    };
    const std::string head = git("rev-parse HEAD");
    const bool dirty = !git("status --porcelain --untracked-files=no").empty();
    return dirty ? head + "+changes" : head;
}

}  // namespace

std::string manifest(std::size_t records) {
    // What every translation here depends on: a reader refuses an export made for another host
    // layout.
    std::string m = "format gpu-x2w-1\nkey blake3-16 of the D3D12_SHADER_BYTECODE bytes\n";
    m += "producer x2w-shaders\n";
    // Hash x2w::vs::translation_fingerprint(), which includes the host settings.
    m += "build " + build_hex() + "\n";
    m += "commit " + commit() + "\n";
    for (const auto& [name, value] : settings()) m += name + " " + value + "\n";
    m += "cache none: every container is translated by X2W's translator, so nothing is cached or missing and there "
         "are no orphan entries\n";
    m += "records " + u(records) + "\n";
    return m;
}

int shaders_main(const std::vector<std::string>& args) {
    DumpOptions o;
    bool have_out = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--out" && i + 1 < args.size()) {
            o.out = std::filesystem::path(std::u8string(args[i + 1].begin(), args[i + 1].end())); ++i;
            have_out = true;
        } else if (a == "--jobs" && i + 1 < args.size()) {
            const std::string& v = args[++i];
            char* end = nullptr;
            const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
            if (v.empty() || *end) {
                std::cerr << "x2w shaders: --jobs takes a number\n";
                return 2;
            }
            o.jobs = static_cast<std::size_t>(n);
        } else if (a.rfind("--", 0) == 0) {
            std::cerr << "x2w shaders: unknown option " << a << "\n";
            return 2;
        } else {
            o.paths.push_back(std::filesystem::path(std::u8string(a.begin(), a.end())));
        }
    }
    if (!have_out || o.paths.empty()) {
        std::cerr << "usage: x2w shaders <paths...> --out <dir> [--jobs N]\n";
        return 2;
    }
    std::cerr << probe_host() << "\n";
    o.build = build_hex();
    o.manifest = [](std::size_t records) { return manifest(records); };
    try {
        const DumpCounts c = dump(o, make);
        std::cout << c.summary;
    } catch (const std::exception& e) {
        std::cerr << "x2w shaders: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace x2w::tool
