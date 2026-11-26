#pragma once

// Helpers shared by the card tests of the upstream Rust emitter (waterfall_tests.rs, wave64_tests.rs,
// permlane_tests.rs, cube_lod_tests.rs, probe_tests.rs, size_tests.rs): the front end run over a
// synthetic NativeProgram, the HostShader made from what lowering gives, the validator, and the
// host (hardware device and WARP).
//
// Every use of the host and validator ports (vs-emit's host.h / validator.h) goes through the
// wrappers here. Uses of vs-front's test fixtures are marked `// FRONT?`.
//
// Error mapping: Rust `.expect(..)` / `panic!` / `assert!` failing inside a test abort that test;
// here they throw (TestFail via `fail` / `require`, or the ShaderError / host error as is), and
// `guarded` in each file's main counts the test failed and goes on to the next, as cargo test does.
//
// Dumps: each test calls `x2w::vs::lower::test::set_dump_name("emit__<module>__<fn>")` first, the
// Rust thread name with `::` as `__`, so VS_DUMP files line up with GMR/vs_port/lower/dump_rs.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/host.h"      vs-emit: x2w::vs::host (host.rs)
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/shader.h"  // set_raw_views_here
#include "x2w_shader/emit/validator.h" vs-emit: x2w::vs::validator (validator.rs)
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"
#include "test_util.h"

namespace card {

namespace vs = ::x2w::vs;
using vs::HostShader;
using vs::ShaderError;
using vs::UserData;
using vs::lower::Lowered;
using vs::lower::Parts;
using vs::native::ComputeIo;
using vs::native::HwStage;
using vs::native::NativeProgram;
using vs::native::ResourceType;
using vs::native::RootParam;
using vs::native::RootType;
using vs::native::StageIo;
using vs::native::UsedResource;

// ---- failures ----------------------------------------------------------------------------------

// A Rust panic (`expect`, `panic!`, a failed `assert!`) inside a test.
struct TestFail : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string& what) { throw TestFail(what); }

// Rust `assert!(cond, msg)`: stops the test.
inline void require(bool ok, const std::string& what) {
    if (!ok) fail(what);
}

// Runs one test: a throw counts as a failure (Rust: the test panicked), and the next test runs.
template <class F>
void guarded(const char* name, F&& f) {
    try {
        f();
    } catch (const ShaderError& e) {
        std::fprintf(stderr, "%s: ShaderError: %s\n", name, e.to_string().c_str());
        ++test_util::failures();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        ++test_util::failures();
    }
}

// Rust `matches!(r, Err(ShaderError::Unsupported { what, .. }) if what == want)`.
inline bool is_unsupported(const ShaderError& e, std::string_view want) {
    return e.kind == ShaderError::Kind::Unsupported && std::string_view(e.reason) == want;
}

// ---- words -------------------------------------------------------------------------------------

// Rust `words.iter().flat_map(|w| w.to_le_bytes()).collect()`.
inline std::vector<std::uint8_t> le_bytes(std::span<const std::uint32_t> words) {
    std::vector<std::uint8_t> code;
    code.reserve(words.size() * 4);
    for (std::uint32_t w : words)
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    return code;
}

// Rust `code.chunks_exact(4).map(u32::from_le_bytes).collect()`.
inline std::vector<std::uint32_t> le_words(std::span<const std::uint8_t> code) {
    std::vector<std::uint32_t> words;
    for (std::size_t i = 0; i + 4 <= code.size(); i += 4) words.push_back(test_util::u32le(code, i));
    return words;
}

// Rust `[a, b, c].concat()` of word slices.
inline std::vector<std::uint32_t> concat(std::initializer_list<std::span<const std::uint32_t>> parts) {
    std::vector<std::uint32_t> out;
    for (auto p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

inline std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

inline float f32_from_bits(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Rust `{:#x}`.
inline std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

// ---- the front end and lowering ----------------------------------------------------------------

// Rust `lower::TEST_HOST_WAVE.replace(Some(n))` ... `.set(was)`: the tests' `was` is always None
// (nothing else sets it on the test thread), so the guard restores None.
struct HostWave {
    explicit HostWave(std::optional<std::uint32_t> lanes) { vs::lower::test::set_host_wave(lanes); }
    ~HostWave() { vs::lower::test::set_host_wave(std::nullopt); }
    HostWave(const HostWave&) = delete;
    HostWave& operator=(const HostWave&) = delete;
};

// Rust `lower::TEST_F16_NEAREST.replace(Some(b))` ... `.set(was)` (was: None).
struct F16Nearest {
    explicit F16Nearest(std::optional<bool> on) { vs::lower::test::set_f16_nearest(on); }
    ~F16Nearest() { vs::lower::test::set_f16_nearest(std::nullopt); }
    F16Nearest(const F16Nearest&) = delete;
    F16Nearest& operator=(const F16Nearest&) = delete;
};

// Rust `super::shader::RAW_VIEWS_HERE.set(Some(n))` ... `.set(None)`.
struct RawViewsHere {
    explicit RawViewsHere(std::optional<std::uint32_t> first) { vs::shader::set_raw_views_here(first); }
    ~RawViewsHere() { vs::shader::set_raw_views_here(std::nullopt); }
    RawViewsHere(const RawViewsHere&) = delete;
    RawViewsHere& operator=(const RawViewsHere&) = delete;
};

// The Parts of a compute program: no signature elements, mesh_stride 0.
inline Parts parts_of(const NativeProgram& native, const vs::ir::Program& p, const vs::structure::Structured& st,
                      const vs::io::Io& io, const vs::resource::Resources& res) {
    Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    return parts;
}

// The HostShader of a lowered compute program (Rust's struct literal in each test file).
inline HostShader host_shader(Lowered&& lowered, std::vector<vs::resource::RootIndexedView> root_indexed_views,
                              std::vector<vs::StaticSampler> static_samplers = {}) {
    HostShader shader;
    shader.stage = HwStage::Cs;
    shader.dxil = std::move(lowered.dxil);
    shader.user_data = std::move(lowered.user_data);
    shader.samplers = std::move(lowered.samplers);
    shader.static_samplers = std::move(static_samplers);
    shader.root_cbvs = std::move(lowered.root_cbvs);
    shader.root_srvs = std::move(lowered.root_srvs);
    shader.root_uavs = std::move(lowered.root_uavs);
    shader.root_indexed_views = std::move(root_indexed_views);
    return shader;
}

// ---- the validator ---------------------------------------------------------------------

// Rust `validator::Validator::load(validator::Validator::default_path().as_deref())`: none when
// dxil.dll does not load (the reason in *why).
inline std::optional<vs::validator::Validator> load_validator(std::string* why = nullptr) {
    return vs::validator::Validator::load(vs::validator::Validator::default_path(), why);
}

// Rust `if let Ok(mut v) = Validator::load(..) { v.validate(&mut dxil.clone()).expect(..) }`,
// with `skipped_line`: `else { eprintln!("no dxil.dll; validation skipped") }`. Returns the
// validated copy (the validator rewrites the container hash), or none without a validator.
inline std::optional<std::vector<std::uint8_t>> validate(const HostShader& shader, bool skipped_line = false) {
    std::string why;
    auto v = load_validator(&why);
    if (!v) {
        if (skipped_line) std::fprintf(stderr, "no dxil.dll; validation skipped\n");
        return std::nullopt;
    }
    std::vector<std::uint8_t> copy = shader.dxil;
    if (auto rejected = v->validate(copy)) fail("DXIL validation failed: " + *rejected);
    return copy;
}

// ---- the host --------------------------------------------------------------------------

using Host = vs::host::Host;
using View = vs::host::View;  // Rust enum View<'a> (borrows its data)
using Stride = vs::host::Stride;
using Root = vs::host::Root;
using Texel = std::array<float, 4>;

// Rust `View::Texture(w, h, &texels)`.
inline View texture(std::uint32_t w, std::uint32_t h, std::span<const Texel> texels) {
    return View::texture(w, h, texels);
}
// Rust `View::Cube(size, mips, &texels)`.
inline View cube(std::uint32_t size, std::uint16_t mips, std::span<const Texel> texels) {
    return View::cube(size, mips, texels);
}
// Rust `View::RawUav(dwords)`.
inline View raw_uav(std::uint32_t dwords) { return View::raw_uav(dwords); }
// Rust `View::UintTextureUav(w, h)`.
inline View uint_texture_uav(std::uint32_t w, std::uint32_t h) { return View::uint_texture_uav(w, h); }
// Rust `View::Buffer { data, stride }`.
inline View buffer(std::span<const std::uint32_t> data, Stride stride) { return View::buffer(data, stride); }
// Rust `Stride::Raw` / `Stride::Structured(n)`.
inline Stride stride_raw() { return Stride::raw(); }
inline Stride stride_structured(std::uint32_t bytes) { return Stride::structured(bytes); }

// Rust `Host::new(warp)`: none (and `why`) without an SM 6.6 device. Host::create(bool)
// throws host::Error (Rust Result<Self, String>).
inline std::optional<Host> new_host(bool warp, std::string* why = nullptr) {
    try {
        return Host::create(warp);
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return std::nullopt;
    }
}

// What `Host::compute_pipeline` gives: Rust `(ID3D12RootSignature, ID3D12PipelineState)`.
using Pipeline = vs::host::Pipeline;

// Rust `host.compute_pipeline(&shader)`: none (and `why`) when the device refuses it.
// compute_pipeline(Stage) throws host::Error.
inline std::optional<Pipeline> compute_pipeline(const Host& host, const HostShader& shader, std::string* why = nullptr) {
    try {
        return host.compute_pipeline(shader);
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return std::nullopt;
    }
}

// Rust `host.dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views,
// samplers, groups }).unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"))`: each UAV's
// dwords in heap order.
inline std::vector<std::vector<std::uint32_t>> dispatch(const Host& host, const Pipeline& pipe,
                                                        std::vector<std::uint32_t> constants,
                                                        std::span<const View> views,
                                                        std::span<const D3D12_SAMPLER_DESC> samplers,
                                                        std::array<std::uint32_t, 3> groups,
                                                        const std::string& label = "") {
    vs::host::Dispatch d;
    d.roots.emplace_back(0u, Root::constants(std::move(constants)));
    d.views = views;
    d.samplers = samplers;
    d.groups = groups;
    try {
        const auto& [root, pso] = pipe;
        return host.dispatch(root.Get(), pso.Get(), d);
    } catch (const TestFail&) {
        throw;
    } catch (const std::exception& e) {
        fail((label.empty() ? std::string() : label + ": ") + "dispatch failed: " + e.what());
    }
}

// Rust `options.WaveLaneCountMin`, `options.WaveLaneCountMax` of
// D3D12_FEATURE_D3D12_OPTIONS1 (permlane_tests.rs: `host.device.CheckFeatureSupport`).
// host.device: a public ID3D12Device (ComPtr or windows-crate-like wrapper with ->).
inline std::pair<std::uint32_t, std::uint32_t> wave_lane_counts(const Host& host) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options{};
    HRESULT hr = host.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options,
                                                  static_cast<UINT>(sizeof options));
    if (FAILED(hr)) fail("wave capabilities");
    return {options.WaveLaneCountMin, options.WaveLaneCountMax};
}

// Rust `super::valu_tests::ONE_AT_A_TIME`: one GPU test at a time (here each test file is an
// executable running its tests in order, so the lock never waits; kept to mirror the Rust).
inline std::mutex& one_at_a_time() {
    static std::mutex m;
    return m;
}

// ---- waterfall_tests::run ----------------------------------------------------------------------

// Rust waterfall_tests::run: runs `shader` with `views` at heap indices 0.. (the UAV last) on the
// hardware GPU and on WARP (each skipped without a device) and returns, per one that ran, the
// first UAV's dwords.
inline std::vector<std::pair<const char*, std::vector<std::uint32_t>>> run(const HostShader& shader,
                                                                           std::span<const View> views) {
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.MaxLOD = std::numeric_limits<float>::max();
    const D3D12_SAMPLER_DESC samplers[] = {sampler};
    std::vector<std::pair<const char*, std::vector<std::uint32_t>>> out;
    std::lock_guard one(one_at_a_time());
    const std::pair<const char*, bool> devices[] = {{"hardware", false}, {"warp", true}};
    for (auto [label, warp] : devices) {
        std::string why;
        auto host = new_host(warp, &why);
        if (!host) {
            std::fprintf(stderr, "no SM 6.6 %s device (%s); skipped\n", label, why.c_str());
            continue;
        }
        auto pipe = compute_pipeline(*host, shader, &why);
        if (!pipe) fail(std::string(label) + ": compute_pipeline failed: " + why);
        // Both tables start at heap index 0.
        std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
        for (const UserData& u : shader.user_data) {
            require(u.root == RootType::Table(), "user data not a table");
            ud[u.index] = 0;
        }
        auto words = dispatch(*host, *pipe, std::move(ud), views, samplers, {1, 1, 1}, label);
        out.emplace_back(label, words[0]);
    }
    return out;
}

// ---- intersect_tests::build_compute_shader -----------------------------------------------------

// Rust intersect_tests::build_compute_shader: `native`, a compute program, translated (no
// container). Throws ShaderError from resource::track and lowering (Rust `?`); the other stages'
// errors are the Rust `.expect(..)`s (they throw too).
inline std::pair<HostShader, vs::resource::Resources> build_compute_shader(const NativeProgram& native) {
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, native.vgprs);
    vs::structure::Structured st = vs::structure::structure(p);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered lowered = vs::lower::lower(parts_of(native, p, st, io, res));
    HostShader shader = host_shader(std::move(lowered), res.root_indexed_views);
    validate(shader);
    return {std::move(shader), std::move(res)};
}

// ---- programs ----------------------------------------------------------------------------------

// RootParam { root, table_offset_dwords, sgprs, base_register, user_data, slot }.
inline RootParam root_param(RootType root, std::uint32_t table_offset_dwords, std::uint32_t sgprs,
                            std::uint32_t base_register, std::uint8_t user_data, std::uint8_t slot) {
    RootParam r;
    r.root = root;
    r.table_offset_dwords = table_offset_dwords;
    r.sgprs = sgprs;
    r.base_register = base_register;
    r.user_data = user_data;
    r.slot = slot;
    return r;
}

// UsedResource { kind, root, table_offset_dwords, lower_bound, stride, dwords, user_data, slot }.
inline UsedResource used_resource(ResourceType kind, RootType root, std::uint32_t table_offset_dwords,
                                  std::uint32_t lower_bound, std::uint32_t stride, std::uint8_t dwords,
                                  std::uint8_t user_data, std::uint8_t slot) {
    UsedResource u;
    u.kind = kind;
    u.root = root;
    u.table_offset_dwords = table_offset_dwords;
    u.lower_bound = lower_bound;
    u.stride = stride;
    u.dwords = dwords;
    u.user_data = user_data;
    u.slot = slot;
    return u;
}

// StageIo::Cs(ComputeIo { rsrc2, threads }).
inline StageIo compute_io(std::uint32_t rsrc2, std::array<std::uint32_t, 3> threads) {
    ComputeIo cs;
    cs.rsrc2 = rsrc2;
    cs.threads = threads;
    return cs;
}

// Rust `p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).count()`.
inline std::size_t count_insts(const vs::ir::Program& p, std::string_view name) {
    std::size_t n = 0;
    for (const auto& b : p.blocks)
        for (const auto& i : b.insts)
            if (i.name() == name) ++n;
    return n;
}

}  // namespace card
