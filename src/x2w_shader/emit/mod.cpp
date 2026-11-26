// the entry points (translate, emit), the wave64 retries, emit_program and the translation fingerprint. The
// settings are in settings.cpp, the signature readers in mod_signature.cpp.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <string_view>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/opt.h"
#include "x2w_shader/emit/threading.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/spill.h"
#include "x2w_shader/structure.h"

#ifdef X2W_VS_FINGERPRINT_HEADER
#include "x2w_vs_fingerprint.h"  // generated at build time (cmake/x_translate.cmake)
#endif
#ifndef X2W_VS_SOURCE_HASH
#define X2W_VS_SOURCE_HASH "unknown"
#endif
#ifndef X2W_VS_SWITCHES
#define X2W_VS_SWITCHES ""
#endif

namespace x2w::vs {

namespace {

using dxbc::Container;
using dxbc::FourCC;
using native::HwStage;
using native::NativeProgram;

std::optional<std::string_view> env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string_view(v);
}

bool is_unsupported(const ShaderError& e) { return e.kind == ShaderError::Kind::Unsupported; }
bool what_is(const ShaderError& e, const char* what) {
    return is_unsupported(e) && std::strcmp(e.reason, what) == 0;
}

struct Emitted {
    std::optional<std::pair<HostShader, std::optional<std::uint32_t>>> ok;
    std::optional<ShaderError> err;
    bool is_ok() const { return ok.has_value(); }
    // Ok with a wave size other than 64.
    bool done() const { return ok && ok->second != std::optional<std::uint32_t>(64); }
};

template <class F>
Emitted capture(F&& f) {
    Emitted r;
    try {
        r.ok = f();
    } catch (const ShaderError& e) {
        r.err = e;
    }
    return r;
}

HostShader unwrap(Emitted r) {
    if (r.err) throw *r.err;
    return std::move(r.ok->first);
}

// The translated program, and the wave size it requires of the host.
std::pair<HostShader, std::optional<std::uint32_t>> emit_program(const Container& c, const NativeProgram& native,
                                                                 const ir::Program& p, const structure::Structured& st,
                                                                 const io::Io& io, const resource::Resources& res) {
    auto sig = [&](FourCC fourcc) -> std::vector<container::SigElement> {
        if (const dxbc::Part* part = c.part(fourcc)) return signature(part->data);
        return {};
    };
    const bool pixel = native.stage == HwStage::Ps;
    std::vector<container::SigElement> inputs = sig(FourCC::ISG1);
    std::vector<container::SigElement> outputs = sig(FourCC::OSG1);
    // Interpolation as DXC declares it by default; pixel inputs that are read get the mode their
    // barycentrics say.
    std::vector<container::SigElement>& interpolated = pixel ? inputs : outputs;
    for (container::SigElement& e : interpolated) {
        if (e.kind == 3)
            e.interp = 4;
        else if (e.kind == 13)
            // Constant, as DXC declares it (the AMD compiler crashed on a front face without a
            // mode).
            e.interp = 1;
        else if (e.kind == 28)
            // Until a read says which mode (lower).
            e.interp = 2;
        else if ((e.kind == 0 || e.kind == 6 || e.kind == 7) && e.comp == 9)
            e.interp = 2;
        else if (e.kind == 0)
            e.interp = 1;
        else
            e.interp = 0;
    }
    if (native.stage == HwStage::Vs) {
        if (const dxbc::Part* psv = c.part(FourCC::PSV0)) vertex_output_interpolation(psv->data, outputs);
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL wants the position
    // centroid (or per sample), as DXC declares it then.
    if (pixel && std::any_of(outputs.begin(), outputs.end(),
                             [](const container::SigElement& e) { return e.kind == 18 || e.kind == 19; })) {
        for (container::SigElement& e : inputs)
            if (e.kind == 3) e.interp = 5;
    }
    // Per-primitive outputs are constant. The console never exports a primitive id or cull flag a
    // mesh shader writes (it culls with the primitive export), so those rows are left out.
    std::vector<container::SigElement> prims = sig(FourCC::PSG1);
    prims.erase(std::remove_if(prims.begin(), prims.end(),
                               [](const container::SigElement& e) { return e.kind == 10 || e.kind == 30; }),
                prims.end());
    for (container::SigElement& e : prims) e.interp = 1;
    // A mesh group's id arrives multiplied by max(threads, vertices, primitives) (the code divides
    // by that); the thread count is in PSV0 from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right when the HLSL group
    // fills its waves (the title's do).
    std::uint32_t mesh_stride = 0;
    if (const auto* ms = std::get_if<native::MeshIo>(&native.io)) {
        std::optional<std::uint32_t> threads;
        if (const dxbc::Part* psv = c.part(FourCC::PSV0)) {
            auto n = try_shader([&] { return dxbc::u32_at(psv->data, 0); });
            if (n && *n >= 48) threads = try_shader([&] { return dxbc::u32_at(psv->data, 4 + 36); });
        }
        mesh_stride = threads ? std::max({*threads, ms->max_vertices, ms->max_primitives}) : ms->threads;
    }
    lower::Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.inputs = std::move(inputs);
    parts.outputs = std::move(outputs);
    parts.prims = std::move(prims);
    parts.mesh_stride = mesh_stride;
    lower::Lowered l = lower::lower(parts);
    const std::optional<std::uint32_t> wave_size = l.wave_size;
    HostShader shader;
    shader.stage = native.stage;
    shader.dxil = std::move(l.dxil);
    shader.user_data = std::move(l.user_data);
    for (std::size_t n = 0; n < l.samplers.size(); ++n) {
        StaticSampler s = StaticSampler::from_words(l.samplers[n]);
        if (((l.cmp_samplers >> n) & 1) != 0) s.filter |= 0x80;
        shader.static_samplers.push_back(s);
    }
    shader.samplers = std::move(l.samplers);
    shader.root_cbvs = std::move(l.root_cbvs);
    shader.root_srvs = std::move(l.root_srvs);
    shader.root_uavs = std::move(l.root_uavs);
    shader.root_indexed_views = res.root_indexed_views;
    return {std::move(shader), wave_size};
}

// p (a console wave64 compute program on a host of 32-lane waves) with the per-lane branches
// around its cross-lane code taken wave-wide (structure::structure_wide): the lowering names the
// blocks where the two host waves must meet and could not (lower::take_needs), the branches whose
// regions hold them are widened, and so on until it translates or a branch cannot be widened.
// Only where a thread group is one console wave.
Emitted widen(const Container& c, const NativeProgram& native, const ir::Program& p, bool why) {
    const ShaderError NOT =
        ShaderError::unsupported(0, "wave-wide branches in a thread group of several console waves");
    const auto* cs = std::get_if<native::ComputeIo>(&native.io);
    if (!cs) return Emitted{std::nullopt, NOT};
    // (a program whose waves are the host's meets no one: no barrier, any group)
    const bool own = !(native.wave64 && lower::host_max_wave() == 32);
    if (!own && cs->threads[0] * cs->threads[1] * cs->threads[2] != 64) {
        if (why)
            std::fprintf(stderr, "WAVE64-WIDE not tried: thread groups of [%u, %u, %u]\n", cs->threads[0], cs->threads[1],
                         cs->threads[2]);
        return Emitted{std::nullopt, NOT};
    }
    std::set<std::size_t> wide;
    for (;;) {
        Emitted r;
        std::vector<std::size_t> needs;
        std::optional<structure::Structured> st;
        try {
            st = structure::structure_wide(p, wide);
            const io::Io io = io::raise(p, native);
            const resource::Resources res = resource::track(p, native.layout, native.stage, &*st);
            lower::take_needs();
            r = capture([&] { return emit_program(c, native, p, *st, io, res); });
            needs = lower::take_needs();
        } catch (const ShaderError& e) {
            return Emitted{std::nullopt, e};
        }
        // (or translated for 64-lane waves still: shared memory read and written where not every
        // lane runs, whose blocks the lowering names too)
        const bool stuck =
            (r.err && is_unsupported(*r.err) &&
             (lower::needs_every_lane(r.err->reason) || (std::strcmp(r.err->reason, lower::LANE_MASK) == 0 && !needs.empty()))) ||
            (r.ok && r.ok->second == std::optional<std::uint32_t>(64) && !needs.empty());
        if (!stuck) {
            if (why) std::fprintf(stderr, "WAVE64-WIDE %zu branches taken wave-wide\n", wide.size());
            return r;
        }
        std::vector<std::size_t> more;
        for (std::size_t b : lower::branches_around(*st, needs).value_or(std::vector<std::size_t>{}))
            if (!wide.count(b)) more.push_back(b);
        if (more.empty()) {
            if (why)
                std::fprintf(stderr, "WAVE64-WIDE stopped with %zu branches wave-wide: a branch around it cannot be\n",
                             wide.size());
            return r;
        }
        wide.insert(more.begin(), more.end());
    }
}

void report(bool why, const std::string& what, const Emitted& r) {
    if (!why) return;
    if (r.ok) {
        if (r.ok->second)
            std::fprintf(stderr, "WAVE64-RETRY %s: wave Some(%u)\n", what.c_str(), *r.ok->second);
        else
            std::fprintf(stderr, "WAVE64-RETRY %s: wave None\n", what.c_str());
    } else {
        std::fprintf(stderr, "WAVE64-RETRY %s: %s\n", what.c_str(), r.err->to_string().c_str());
    }
}

// The program in another form, with its earlier stages run again.
struct Form {
    ir::Program p;
    structure::Structured st;
    io::Io io;
    resource::Resources res;
};

std::optional<Form> form(const NativeProgram& native, std::optional<ir::Program> p) {
    if (!p) return std::nullopt;
    auto st = try_shader([&] { return structure::structure(*p); });
    if (!st) return std::nullopt;
    auto io = try_shader([&] { return io::raise(*p, native); });
    if (!io) return std::nullopt;
    auto res = try_shader([&] { return resource::track(*p, native.layout, native.stage, &*st); });
    if (!res) return std::nullopt;
    return Form{std::move(*p), std::move(*st), std::move(*io), std::move(*res)};
}

HostShader translate_plain(std::span<const std::uint8_t> container) {
    const Container c = Container::parse(container);
    const dxbc::Part* dxil = c.part(FourCC::DXIL);
    if (!dxil) dxil = c.part(FourCC::ACCC);
    if (!dxil) throw ShaderError::malformed("no DXIL part");
    std::vector<NativeProgram> programs = native::native_programs(dxil->data);
    auto it = std::find_if(programs.begin(), programs.end(), [](const NativeProgram& p) {
        return p.stage == HwStage::Vs || p.stage == HwStage::Ps || p.stage == HwStage::Cs || p.stage == HwStage::Ms;
    });
    if (it == programs.end())
        throw ShaderError::unsupported(0, "a container without a vertex, pixel, compute or mesh program");
    const NativeProgram& native = *it;
    const ir::Program p = spill::build(native);
    // Structurisation first: tracing per-lane descriptors needs it (its failure is reported after
    // the earlier stages').
    std::optional<structure::Structured> st;
    std::optional<ShaderError> st_err;
    try {
        st = structure::structure(p);
    } catch (const ShaderError& e) {
        st_err = e;
    }
    const resource::Resources res = resource::track(p, native.layout, native.stage, st ? &*st : nullptr);
    const io::Io io = io::raise(p, native);
    if (st_err) throw *st_err;
    return emit(c, native, p, *st, io, res);
}

}  // namespace

std::uint32_t HostShader::user_data_dwords() const {
    std::uint32_t n = 0;
    for (const UserData& u : user_data) n = std::max(n, u.index + 1);
    return (n + 3) / 4 * 4;
}

std::vector<std::uint8_t> translation_fingerprint() {
    std::vector<std::uint8_t> out;
    auto bytes = [&](std::string_view s) { out.insert(out.end(), s.begin(), s.end()); };
    auto le32 = [&](std::uint32_t v) {
        for (int k = 0; k < 4; ++k) out.push_back(std::uint8_t(v >> (8 * k)));
    };
    bytes(X2W_VS_SOURCE_HASH);
#ifdef X2W_VS_DIAGNOSTICS
    out.push_back(1);
#else
    out.push_back(0);
#endif
#ifdef NDEBUG
    out.push_back(0);
#else
    out.push_back(1);
#endif
    std::string_view names = X2W_VS_SWITCHES;
    while (!names.empty()) {
        std::size_t comma = names.find(',');
        std::string_view name = names.substr(0, comma);
        names = comma == std::string_view::npos ? std::string_view() : names.substr(comma + 1);
        if (name.empty()) continue;
        bytes(name);
        out.push_back('=');
        const std::string n(name);
        if (auto v = env(n.c_str())) bytes(*v);
        out.push_back(0);
    }
    le32(lower::host_max_wave());
    le32(lower::host_rdna());
    le32(shader::view_heap_limit());
    le32(shader::raw_views());
    le32(loop_limit());
    out.push_back(group_sync());
    le32(shader_opt());
    return out;
}

HostShader translate(std::span<const std::uint8_t> container) {
    try {
        return translate_plain(container);
    } catch (const ShaderError& e) {
        if (!what_is(e, lower::FIRST_HALF)) throw;
    }
    return lower::with_first_half_pairs([&] { return translate_plain(container); });
}

HostShader emit(const dxbc::Container& c, const native::NativeProgram& native, const ir::Program& p,
                const structure::Structured& st, const io::Io& io, const resource::Resources& res) {
    // (what an earlier program's lowering on this thread left behind)
    lower::take_needs();
    Emitted original = capture([&] { return emit_program(c, native, p, st, io, res); });
    // A console wave64 compute program on a host of 32-lane waves that still needs 64 lanes, or
    // whose cross-lane code sits where not every lane runs: once more with its early-outs left out
    // (ir::without_early_outs), which is what usually keeps the whole body from being a block
    // every lane runs.
    bool stuck = false;
    if (original.ok)
        stuck = original.ok->second == std::optional<std::uint32_t>(64);
    else if (is_unsupported(*original.err))
        stuck = lower::needs_every_lane(original.err->reason);
    // (a mask with lanes elsewhere is only helped by the exact forms: not by each half-wave on its
    // own)
    const bool exact_only = original.err && what_is(*original.err, lower::MASK_ELSEWHERE);
    if (stuck && native.stage == HwStage::Cs && native.wave64 && lower::host_max_wave() == 32) {
        const bool why = env("X2W_SHADERCFG_WAVE_WHY").has_value();
        // Each host wave as a wave of its own (lower::with_local_waves), for programs still stopped
        // where not every lane runs.
        const bool local_on = [] {
            auto v = env("X2W_SHADERCFG_WAVE_LOCAL");
            return !v || *v != "0";
        }();
        // Where the wave operations are the allocation idiom (lower::wave_local) that is exact.
        // Other programs are taken the same way last of all, each half-wave giving its own results
        // (approximate); X2W_SHADERCFG_WAVE_LOCAL=idiom refuses those instead.
        const bool strict = [] {
            auto v = env("X2W_SHADERCFG_WAVE_LOCAL");
            return v && *v == "idiom";
        }();
        auto idiom = [&](const ir::Program& fp, const structure::Structured& fst) {
            auto err = lower::wave_local(fp, fst);
            if (!err) return true;
            if (why)
                std::fprintf(stderr, "WAVE64-RETRY local waves: not the allocation idiom at %s: %s\n",
                             hex_alt(err->first).c_str(), err->second);
            return false;
        };
        // The program in two other forms: without its early-outs, and without any branch that only
        // skips code no lane is left for.
        std::optional<Form> f_early = form(native, ir::without_early_outs(p));
        std::optional<Form> f_skips = form(native, ir::without_skips(p, false));
        const std::pair<const char*, std::optional<Form>*> forms[2] = {{"without early-outs", &f_early},
                                                                      {"without skips", &f_skips}};
        for (const auto& [name, f] : forms) {
            if (!*f) continue;
            const Form& g = **f;
            Emitted again = capture([&] { return emit_program(c, native, g.p, g.st, g.io, g.res); });
            report(why, name, again);
            if (again.done()) return unwrap(std::move(again));
        }
        const bool wide_on = [] {
            auto v = env("X2W_SHADERCFG_WAVE_WIDE");
            return v && *v == "1";
        }();
        // exact first: each host wave on its own where that is exact (the idiom, every form; no
        // barriers), then the other exact forms; then approximate
        for (const bool exact : {true, false}) {
            if (!exact) {
                // Before anything approximate, the exact forms that cost group barriers. Every skip
                // left out that the code after it cannot see (ir::without_skips, deep).
                const bool deep_on = [] {
                    auto v = env("X2W_SHADERCFG_WAVE_DEEP_SKIPS");
                    return !v || *v != "0";
                }();
                std::optional<Form> deep = deep_on ? form(native, ir::without_skips(p, true)) : std::nullopt;
                if (deep) {
                    Emitted again = capture([&] { return emit_program(c, native, deep->p, deep->st, deep->io, deep->res); });
                    report(why, "without skips, scalars followed", again);
                    if (again.done()) return unwrap(std::move(again));
                }
                // Still stopped where not every lane runs: the per-lane branches around those places
                // taken as the console takes them, by the whole wave at once (widen), at two group
                // barriers per such branch.
                if (wide_on) {
                    const ir::Program* base = deep ? &deep->p : f_skips ? &f_skips->p : f_early ? &f_early->p : &p;
                    Emitted again = widen(c, native, *base, why);
                    report(why, "wave-wide branches", again);
                    if (again.done()) return unwrap(std::move(again));
                }
                if (strict) break;
            }
            if (!local_on || exact_only) continue;
            for (const auto& [name, f] : forms) {
                if (!*f) continue;
                const Form& g = **f;
                if (idiom(g.p, g.st) != exact) continue;
                Emitted local = capture(
                    [&] { return lower::with_local_waves([&] { return emit_program(c, native, g.p, g.st, g.io, g.res); }); });
                report(why, std::string("local waves (") + (exact ? "idiom" : "APPROXIMATE") + "), " + name, local);
                if (local.done()) return unwrap(std::move(local));
            }
            if (idiom(p, st) == exact) {
                Emitted local =
                    capture([&] { return lower::with_local_waves([&] { return emit_program(c, native, p, st, io, res); }); });
                report(why, exact ? "local waves (idiom)" : "local waves (APPROXIMATE)", local);
                if (local.done()) return unwrap(std::move(local));
            }
        }
        if (local_on && !exact_only) {
            // Last: every mask and lane read per host wave.
            if (!strict) {
                for (const auto& [name, f] : forms) {
                    if (!*f) continue;
                    const Form& g = **f;
                    Emitted local = capture([&] {
                        return lower::with_local_waves_any([&] { return emit_program(c, native, g.p, g.st, g.io, g.res); });
                    });
                    report(why, std::string("local waves (APPROXIMATE, any mask), ") + name, local);
                    if (local.done()) return unwrap(std::move(local));
                }
                Emitted local = capture(
                    [&] { return lower::with_local_waves_any([&] { return emit_program(c, native, p, st, io, res); }); });
                report(why, "local waves (APPROXIMATE, any mask)", local);
                if (local.done()) return unwrap(std::move(local));
            }
        }
    }
    // SCC of a mask operation joined with a scalar test behind a per-lane skip (the 32-lane trace
    // programs): right once the wave takes that skip together (widen; the lowering has named the
    // skipped blocks).
    if (original.err && native.stage == HwStage::Cs && what_is(*original.err, lower::LANE_MASK)) {
        auto v = env("X2W_SHADERCFG_WAVE_WIDE");
        if (!v || *v != "0") {
            Emitted again = widen(c, native, p, env("X2W_SHADERCFG_WAVE_WHY").has_value());
            if (again.ok)
                original = std::move(again);
            else
                original = capture([&] { return emit_program(c, native, p, st, io, res); });
        }
    }
    // Retry unsupported programs only. Threading can separate an unknown register value from its
    // apparent use on a mutually exclusive path.
    if (!(original.err && is_unsupported(*original.err))) return unwrap(std::move(original));
    const auto edges = threading::constant_edges(p, st);
    if (edges.empty()) return unwrap(std::move(original));
    const ir::Program tp = ir::thread_edges(p, edges);
    const structure::Structured tst = structure::structure(tp);
    const io::Io tio = io::raise(tp, native);
    const resource::Resources tres = resource::track(tp, native.layout, native.stage, &tst);
    return emit_program(c, native, tp, tst, tio, tres).first;
}

}  // namespace x2w::vs
