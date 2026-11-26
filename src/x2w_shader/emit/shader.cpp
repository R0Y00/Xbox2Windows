
#include "x2w_shader/emit/shader.h"

#include <algorithm>
#include <string>

#include "x2w_shader/emit/census.h"
#include "x2w_shader/emit/opt.h"
#include "x2w_shader/emit/settings.h"

namespace x2w::vs::shader {

using bitcode::Tag;
using container::Binding;
using container::Psv;

namespace {

std::atomic<std::uint32_t> VIEW_LIMIT{0};
std::atomic<std::uint32_t> RAW_VIEWS{0};
thread_local std::optional<std::uint32_t> RAW_VIEWS_HERE;

}  // namespace

void set_view_heap_limit(std::uint32_t null_slot) { VIEW_LIMIT.store(null_slot, std::memory_order_relaxed); }
std::uint32_t view_heap_limit() { return VIEW_LIMIT.load(std::memory_order_relaxed); }

void set_raw_views(std::uint32_t first) { RAW_VIEWS.store(first, std::memory_order_relaxed); }
std::uint32_t raw_views() {
    if (RAW_VIEWS_HERE) return *RAW_VIEWS_HERE;
    return RAW_VIEWS.load(std::memory_order_relaxed);
}
void set_raw_views_here(std::optional<std::uint32_t> first) { RAW_VIEWS_HERE = first; }

Shader::Shader(bool pixel_) : pixel(pixel_) {
    Ty voidt = m.void_();
    Ty i1 = m.int_(1);
    Ty i8 = m.int_(8);
    Ty i32 = m.int_(32);
    Ty f32 = m.float_();
    Ty i8p = m.ptr(i8);
    Ty handle = m.named_struct("dx.types.Handle", {i8p});
    Ty props = m.named_struct("dx.types.ResourceProperties", {i32, i32});
    Ty bind = m.named_struct("dx.types.ResBind", {i32, i32, i32, i8});
    Ty cbuf_ret = m.named_struct("dx.types.CBufRet.i32", {i32, i32, i32, i32});
    Ty res_ret = m.named_struct("dx.types.ResRet.f32", {f32, f32, f32, f32, i32});
    Ty main_ty = m.func_ty(voidt, {});
    m.ptr(main_ty);
    main_ = m.function("main", main_ty, std::nullopt, true);
    entry = m.new_block();
    t = Types{voidt, i1, i32, f32, handle, props, bind, cbuf_ret, res_ret};
}

V Shader::lds() {
    if (lds_) return *lds_;
    Ty arr = m.array(lds_dwords, t.i32);
    V g = m.global("lds", arr, 3);
    lds_ = g;
    return g;
}

V Shader::op(std::size_t block, std::string_view name, Attr attr, Ty ret, std::uint32_t opcode,
             std::span<const V> args) {
    std::vector<Ty> params = {t.i32};
    for (V a : args) params.push_back(m.type_of(a));
    Ty fty = m.func_ty(ret, std::move(params));
    m.ptr(fty);
    std::string full = "dx.op.";
    full += name;
    V f = m.function(full, fty, attr, false);
    std::vector<V> all = {m.i32(opcode)};
    all.insert(all.end(), args.begin(), args.end());
    return m.call(block, f, std::move(all));
}

V Shader::user_data_handle() {
    V bind_[4] = {m.i32(0), m.i32(0), m.i32(USER_DATA_SPACE), m.i8(2)};  // CBV
    V b = m.aggregate(t.bind, bind_);
    std::size_t block = entry;
    V i = m.i32(0);
    V f = m.bool_(false);
    V h = op(block, "createHandleFromBinding", Attr::ReadNone, t.handle, 217, {b, i, f});
    // The size is known in finish, which fixes this placeholder.
    return annotate(block, h, 13, UINT32_MAX);
}

V Shader::root_cbv_handle(std::uint8_t slot) {
    for (const auto& r : root_cbvs)
        if (r.first == slot) return r.second;
    V r = m.i32(slot);
    V bind_[4] = {r, r, m.i32(ROOT_CBV_SPACE), m.i8(2)};
    V b = m.aggregate(t.bind, bind_);
    V f = m.bool_(false);
    std::size_t block = entry;
    V h = op(block, "createHandleFromBinding", Attr::ReadNone, t.handle, 217, {b, r, f});
    h = annotate(block, h, 13, 65536);
    root_cbvs.emplace_back(slot, h);
    return h;
}

V Shader::root_srv_handle(std::uint8_t slot) {
    for (const auto& r : root_srvs)
        if (r.first == slot) return r.second;
    V r = m.i32(slot);
    V bind_[4] = {r, r, m.i32(ROOT_CBV_SPACE), m.i8(0)};
    V b = m.aggregate(t.bind, bind_);
    V f = m.bool_(false);
    std::size_t block = entry;
    V h = op(block, "createHandleFromBinding", Attr::ReadNone, t.handle, 217, {b, r, f});
    h = annotate(block, h, 11, 0);
    flags |= flags::RAW_BUFFERS;
    root_srvs.emplace_back(slot, h);
    return h;
}

V Shader::root_uav_handle(std::uint8_t slot) {
    for (const auto& r : root_uavs)
        if (r.first == slot) return r.second;
    V r = m.i32(slot);
    V bind_[4] = {r, r, m.i32(ROOT_CBV_SPACE), m.i8(1)};
    V b = m.aggregate(t.bind, bind_);
    V f = m.bool_(false);
    std::size_t block = entry;
    V h = op(block, "createHandleFromBinding", Attr::ReadNone, t.handle, 217, {b, r, f});
    h = annotate(block, h, 11 | 0x1000, 0);
    flags |= flags::RAW_BUFFERS;
    root_uavs.emplace_back(slot, h);
    return h;
}

V Shader::sampler_handle(std::size_t block, std::uint32_t n) {
    V bind_[4] = {m.i32(n), m.i32(n), m.i32(SAMPLER_SPACE), m.i8(3)};  // sampler
    V b = m.aggregate(t.bind, bind_);
    V i = m.i32(n);
    V f = m.bool_(false);
    V h = op(block, "createHandleFromBinding", Attr::ReadNone, t.handle, 217, {b, i, f});
    std::uint32_t cmp = ((cmp_samplers >> n) & 1) != 0 ? 0x8000 : 0;
    return annotate(block, h, 14 | cmp, 0);
}

V Shader::annotate(std::size_t block, V h, std::uint32_t props0, std::uint32_t props1) {
    V p_[2] = {m.i32(props0), m.i32(props1)};
    V p = m.aggregate(t.props, p_);
    return op(block, "annotateHandle", Attr::ReadNone, t.handle, 216, {h, p});
}

V Shader::heap_handle(std::size_t block, V index, bool sampler, bool nonuniform,
                      std::pair<std::uint32_t, std::uint32_t> props) {
    return heap_handle_at(block, index, sampler, nonuniform, props, false);
}

V Shader::heap_handle_at(std::size_t block, V index, bool sampler, bool nonuniform,
                         std::pair<std::uint32_t, std::uint32_t> props, bool bounded) {
    // a view index past the host heap (set_view_heap_limit) reads its null slot instead (the raw
    // twins of buffer views lie past it: such an index comes bounded)
    std::uint32_t limit = bounded ? 0 : VIEW_LIMIT.load(std::memory_order_relaxed);
    // (as dx.op.binary UMin: a compare and select emitted here made the bitcode invalid)
    if (!sampler && limit != 0) {
        V null = m.i32(limit);
        Ty i32t = t.i32;
        V v = op(block, "binary.i32", Attr::ReadNone, i32t, 40, {index, null});
        m.tag(v, Tag::Clamp);
        index = v;
    } else if (sampler) {
        // A sampler heap has at most 2,048 samplers on any host: an index past that (a number
        // read from memory the host has not got yet) stays inside the largest heap there is.
        // shortcut: not the bound heap's own count, which a title heap smaller than 2,048 would
        // need.
        V last = m.i32(2047);
        Ty i32t = t.i32;
        V v = op(block, "binary.i32", Attr::ReadNone, i32t, 40, {index, last});
        m.tag(v, Tag::Clamp);
        index = v;
    }
    flags |= sampler ? flags::SAMPLER_HEAP : flags::RESOURCE_HEAP;
    nonuniform_handles += std::uint32_t(nonuniform);
    V s = m.bool_(sampler);
    V n = m.bool_(nonuniform);
    V h = op(block, "createHandleFromHeap", Attr::ReadNone, t.handle, 218, {index, s, n});
    return annotate(block, h, props.first, props.second);
}

void Shader::fix_user_data_size(std::uint32_t size) {
    V old_[2] = {m.i32(13), m.i32(UINT32_MAX)};
    V old = m.aggregate(t.props, old_);
    V new_[2] = {m.i32(13), m.i32(size)};
    V nw = m.aggregate(t.props, new_);
    m.replace_const(old, nw);
}

std::optional<Md> Shader::signature_md(std::vector<SigElement> elements) {
    if (elements.empty()) return std::nullopt;
    std::vector<std::optional<Md>> list;
    for (std::size_t id = 0; id < elements.size(); ++id) {
        const SigElement& e = elements[id];
        auto c32 = [this](std::uint32_t v) -> std::optional<Md> {
            V c = m.i32(v);
            return m.md_value(c);
        };
        auto c8 = [this](std::uint8_t v) -> std::optional<Md> {
            V c = m.i8(v);
            return m.md_value(c);
        };
        auto idx = c32(e.index);
        std::optional<Md> indices = m.md_node({idx});
        std::optional<Md> usage;
        if (e.usage != 0) {
            auto tag = c32(3);
            auto mask = c32(e.usage);
            usage = m.md_node({tag, mask});
        }
        // Braced lists evaluate from left to right.
        std::vector<std::optional<Md>> ops = {
            c32(std::uint32_t(id)),
            m.md_str(e.name),
            c8(e.comp),
            c8(e.kind),
            indices,
            c8(e.interp),
            c32(1),
            c8(e.cols),
            c32(e.row.value_or(UINT32_MAX)),
            c8(e.row ? e.col : std::uint8_t(0xFF)),
            usage,
        };
        list.push_back(m.md_node(std::move(ops)));
    }
    return m.md_node(std::move(list));
}

std::vector<std::uint8_t> Shader::finish() {
    if (pixel) {
        // Reading a per-vertex value with attributeAtVertex needs a constant input, even when the
        // source declaration was smooth. D3D requires all components packed into the same
        // register to share a mode, including the ones the shader never reads.
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const SigElement& unused = inputs[index];
            if (unused.kind != 0 || unused.usage != 0 || !unused.row) continue;
            std::optional<std::uint8_t> mode;
            bool all = true;
            for (const SigElement& e : inputs) {
                if (!(e.row == unused.row && e.usage != 0)) continue;
                if (!mode)
                    mode = e.interp;
                else if (e.interp != *mode)
                    all = false;
            }
            // Read the first mode before testing the remaining modes.
            if (mode && (*mode == 1 || unused.comp == 9) && all) inputs[index].interp = *mode;
        }
    }
    const std::uint32_t ud_size = (user_data * 4 + 15) / 16 * 16;
    fix_user_data_size(ud_size);
    // The clean-up, before anything is declared about the code: a feature whose last use it
    // removed must not be declared (the validator compares the flags with what the code uses).
    const std::uint32_t passes = shader_opt() & bitcode::opt::MODULE;
    if (passes != 0) {
        m.optimise(passes);
        const bitcode::opt::Usage used = m.usage();
        const std::pair<std::uint64_t, bool> table[] = {
            {flags::ROVS, used.rovs},
            {flags::WAVE_OPS, used.wave_ops},
            {flags::INT64, used.int64},
            {flags::TYPED_UAV_LOADS, used.typed_uav_loads},
            {flags::RESOURCE_HEAP, used.heap_views},
            {flags::SAMPLER_HEAP, used.heap_samplers},
            {flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64, used.atomic64},
        };
        for (const auto& [flag, u] : table)
            if (!u) flags &= ~flag;
    }
    // Unordered views outside pixel and compute programs (memory by address), as DXC counts them: ranges
    // declared below, not views picked from the heap.
    const bool compute = threads.has_value() && !mesh && !geometry;
    if (!root_uavs.empty() && !pixel && !compute) flags |= flags::UAVS_EVERY_STAGE;
    std::optional<Md> resources;
    std::vector<Binding> bindings;
    auto i32c = [this](std::uint32_t v) {
        V c = m.i32(v);
        return m.md_value(c);
    };
    if (user_data > 0 || samplers > 0 || !root_cbvs.empty() || !root_srvs.empty() || !root_uavs.empty()) {
        Md empty = m.md_str("");
        std::optional<Md> cbvs;
        std::vector<std::optional<Md>> cbv_list;
        if (user_data > 0) {
            Ty arr = m.array((user_data + 3) / 4 * 4, t.i32);
            Ty ty = m.named_struct("UserData", {arr});
            Ty p = m.ptr(ty);
            V u = m.undef(p);
            std::vector<std::optional<Md>> ops = {
                i32c(0), m.md_value(u), empty, i32c(USER_DATA_SPACE), i32c(0), i32c(1), i32c(ud_size), std::nullopt,
            };
            cbv_list.push_back(m.md_node(std::move(ops)));
            bindings.push_back(Binding{2, USER_DATA_SPACE, 0, 13});
        }
        Ty blob = m.array(16384, t.i32);
        Ty ty = m.named_struct("RootCbv", {blob});
        Ty p = m.ptr(ty);
        V u = m.undef(p);
        const auto cbv_slots = root_cbvs;
        for (const auto& [slot, h] : cbv_slots) {
            (void)h;
            std::vector<std::optional<Md>> ops = {
                i32c(std::uint32_t(cbv_list.size())),
                m.md_value(u),
                empty,
                i32c(ROOT_CBV_SPACE),
                i32c(slot),
                i32c(1),
                i32c(65536),
                std::nullopt,
            };
            cbv_list.push_back(m.md_node(std::move(ops)));
            bindings.push_back(Binding{2, ROOT_CBV_SPACE, slot, 13});
        }
        if (!cbv_list.empty()) cbvs = m.md_node(std::move(cbv_list));
        std::optional<Md> srvs;
        if (!root_srvs.empty()) {
            Ty sty = m.named_struct("struct.ByteAddressBuffer", {t.i32});
            Ty sp = m.ptr(sty);
            V su = m.undef(sp);
            std::vector<std::optional<Md>> list;
            const auto slots = root_srvs;
            for (std::size_t n = 0; n < slots.size(); ++n) {
                std::uint8_t slot = slots[n].first;
                std::vector<std::optional<Md>> ops = {
                    i32c(std::uint32_t(n)), m.md_value(su), empty, i32c(ROOT_CBV_SPACE), i32c(slot),
                    i32c(1),                i32c(11),       i32c(0), std::nullopt,
                };
                list.push_back(m.md_node(std::move(ops)));
                bindings.push_back(Binding{4, ROOT_CBV_SPACE, slot, 11});
            }
            srvs = m.md_node(std::move(list));
        }
        std::optional<Md> uavs;
        if (!root_uavs.empty()) {
            Ty uty = m.named_struct("struct.RWByteAddressBuffer", {t.i32});
            Ty up = m.ptr(uty);
            V uu = m.undef(up);
            V nov = m.bool_(false);
            Md no = m.md_value(nov);
            std::vector<std::optional<Md>> list;
            const auto slots = root_uavs;
            for (std::size_t n = 0; n < slots.size(); ++n) {
                std::uint8_t slot = slots[n].first;
                // Id, variable, name, space, lower bound, range size, shape (11: raw buffer),
                // globally coherent, counter, rasterizer ordered, no extra properties.
                std::vector<std::optional<Md>> ops = {
                    i32c(std::uint32_t(n)), m.md_value(uu), empty, i32c(ROOT_CBV_SPACE), i32c(slot), i32c(1),
                    i32c(11),               no,             no,    no,                   std::nullopt,
                };
                list.push_back(m.md_node(std::move(ops)));
                bindings.push_back(Binding{7, ROOT_CBV_SPACE, slot, 11});
            }
            uavs = m.md_node(std::move(list));
        }
        std::optional<Md> samps;
        if (samplers > 0) {
            Ty sty = m.named_struct("struct.SamplerState", {t.i32});
            Ty sp = m.ptr(sty);
            V su = m.undef(sp);
            std::vector<std::optional<Md>> list;
            for (std::uint32_t n = 0; n < samplers; ++n) {
                auto kind = std::uint32_t((cmp_samplers >> n) & 1);
                std::vector<std::optional<Md>> ops = {
                    i32c(n), m.md_value(su), empty, i32c(SAMPLER_SPACE), i32c(n), i32c(1),
                    // Sampler kind: 1 comparison.
                    i32c(kind), std::nullopt,
                };
                list.push_back(m.md_node(std::move(ops)));
                bindings.push_back(Binding{1, SAMPLER_SPACE, n, 14});
            }
            samps = m.md_node(std::move(list));
        }
        resources = m.md_node({srvs, uavs, cbvs, samps});
        // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
        auto key = [](const Binding& b) {
            switch (b.kind) {
            case 2: return 0;
            case 1: return 1;
            case 7: return 3;
            default: return 2;
            }
        };
        std::stable_sort(bindings.begin(), bindings.end(),
                         [&](const Binding& a, const Binding& b) { return key(a) < key(b); });
    }
    if (resources) m.named_md("dx.resources", {*resources});

    const bool graphics = !threads || mesh.has_value();
    if (graphics) {
        // shortcut: every output is declared to depend on every input; the exact sets need a
        // dependence pass. Drivers read this state.
        auto scalars = [](const std::vector<SigElement>& els) {
            std::uint32_t mx = 0;
            for (const SigElement& e : els)
                if (e.row) mx = std::max(mx, *e.row * 4 + std::uint32_t(std::uint8_t(e.col + e.cols)));
            return mx;
        };
        const std::uint32_t ni = scalars(inputs), no = scalars(outputs);
        std::vector<std::uint32_t> view = {ni, no};
        // Mesh shaders have no inputs: then the primitive outputs' count.
        if (mesh) view.push_back(scalars(prims));
        for (std::uint32_t i = 0; i < ni; ++i) {
            for (std::uint32_t k = 0; k < (no + 31) / 32; ++k) {
                std::uint32_t bits = std::min(no - 32 * k, 32u);
                view.push_back(bits == 32 ? UINT32_MAX : (1u << bits) - 1);
            }
        }
        // A geometry shader lists four streams: the other three have no outputs.
        if (geometry) view.insert(view.end(), {0, 0, 0});
        Ty arrt = m.array(std::uint32_t(view.size()), t.i32);
        std::vector<V> elems;
        elems.reserve(view.size());
        for (std::uint32_t v : view) elems.push_back(m.i32(v));
        V arr = m.aggregate(arrt, elems);
        Md arrmd = m.md_value(arr);
        Md view_node = m.md_node({arrmd});
        m.named_md("dx.viewIdState", {view_node});
    }

    std::optional<Md> sigs;
    if (graphics) {
        auto ins = signature_md(inputs);
        auto outs = signature_md(outputs);
        auto prs = signature_md(prims);
        sigs = m.md_node({ins, outs, prs});
    }
    // A pixel shader with an SV_StencilRef output (as DXC collects it).
    if (pixel && std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind == 20; }))
        flags |= flags::STENCIL_REF;
    // Entry properties: shader flags (tag 0), geometry state (1), thread-group size (4), wave size
    // (11).
    std::vector<Md> props;
    if (flags != 0) {
        props.push_back(i32c(0));
        Ty i64t = m.int_(64);
        V f = m.int_const(i64t, std::int64_t(flags));
        props.push_back(m.md_value(f));
    }
    if (geometry) {
        // Input primitive (3: triangles), maximum vertices, stream mask, output topology (5:
        // triangle strips), instances.
        props.push_back(i32c(1));
        std::vector<std::optional<Md>> state;
        for (std::uint32_t n : {3u, geometry->max_vertices, 1u, 5u, 1u}) state.push_back(i32c(n));
        props.push_back(m.md_node(std::move(state)));
    }
    if (threads) {
        std::vector<std::optional<Md>> tl;
        for (std::uint32_t n : *threads) tl.push_back(i32c(n));
        Md tn = m.md_node(std::move(tl));
        if (mesh) {
            // Mesh state (tag 9): threads, maximum vertices and primitives, topology, payload
            // bytes.
            props.push_back(i32c(9));
            std::vector<std::optional<Md>> node = {tn};
            for (std::uint32_t n : {mesh->max_vertices, mesh->max_primitives, std::uint32_t(mesh->topology), 0u})
                node.push_back(i32c(n));
            props.push_back(m.md_node(std::move(node)));
        } else {
            props.push_back(i32c(4));
            props.push_back(tn);
        }
    }
    if (wave_size) {
        props.push_back(i32c(11));
        std::optional<Md> n = i32c(*wave_size);
        props.push_back(m.md_node({n}));
    }
    std::optional<Md> propsmd;
    if (!props.empty()) {
        std::vector<std::optional<Md>> list(props.begin(), props.end());
        propsmd = m.md_node(std::move(list));
    }
    Md mainmd = m.md_value(main_);
    Md name = m.md_str("main");
    Md entrymd = m.md_node({mainmd, name, sigs, resources, propsmd});

    Md ident = m.md_str("Xbox2Windows");
    ident = m.md_node({ident});
    m.named_md("llvm.ident", {ident});
    Md one = i32c(1);
    Md six = i32c(6);
    Md version = m.md_node({one, six});
    m.named_md("dx.version", {version});
    m.named_md("dx.valver", {version});
    // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
    const char* kname;
    std::uint32_t kind;
    if (mesh) {
        kname = "ms";
        kind = 13;
    } else if (geometry) {
        kname = "gs";
        kind = 2;
    } else if (threads) {
        kname = "cs";
        kind = 5;
    } else if (pixel) {
        kname = "ps";
        kind = 0;
    } else {
        kname = "vs";
        kind = 1;
    }
    Md smname = m.md_str(kname);
    Md sm = m.md_node({smname, six, six});
    m.named_md("dx.shaderModel", {sm});
    m.named_md("dx.entryPoints", {entrymd});

    const std::vector<std::uint8_t> bc = m.write();
    bitcode::census::note(m, bc);
    std::vector<std::uint8_t> dxil;
    dxil.reserve(24 + bc.size());
    auto put32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
        for (int k = 0; k < 4; ++k) out.push_back(std::uint8_t(v >> (8 * k)));
    };
    put32(dxil, kind << 16 | 0x66);
    put32(dxil, std::uint32_t((24 + bc.size()) / 4));
    dxil.insert(dxil.end(), {'D', 'X', 'I', 'L'});
    put32(dxil, 0x106);
    put32(dxil, 16);
    put32(dxil, std::uint32_t(bc.size()));
    dxil.insert(dxil.end(), bc.begin(), bc.end());

    std::uint64_t features = 0;
    if (flags & flags::DOUBLES) features |= 0x1;
    if (flags & flags::UAVS_EVERY_STAGE) features |= 0x4;
    if (flags & flags::STENCIL_REF) features |= 0x200;
    if (flags & flags::TYPED_UAV_LOADS) features |= 0x800;
    if (flags & flags::ROVS) features |= 0x1000;
    if (flags & flags::WAVE_OPS) features |= 0x4000;
    if (flags & flags::INT64) features |= 0x8000;
    if (flags & flags::BARYCENTRICS) features |= 0x20000;
    if (flags & flags::RESOURCE_HEAP) features |= 0x2000000;
    if (flags & flags::SAMPLER_HEAP) features |= 0x4000000;
    if (flags & flags::ATOMIC_TYPED64) features |= 0x400000;
    if (flags & flags::ATOMIC_HEAP64) features |= 0x10000000;
    std::vector<std::uint8_t> sfi0;
    for (int k = 0; k < 8; ++k) sfi0.push_back(std::uint8_t(features >> (8 * k)));
    const std::vector<std::uint8_t> isg1 = container::signature(inputs, true);
    const std::vector<std::uint8_t> osg1 = container::signature(outputs, false);
    bool flag;
    if (pixel)
        flag = std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind >= 17 && o.kind <= 19; });
    else
        flag = std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind == 3; });
    Psv psvd;
    psvd.stage = std::uint8_t(kind);
    psvd.flag = flag && !threads;
    psvd.inputs = inputs;
    psvd.outputs = outputs;
    psvd.prims = prims;
    if (mesh) {
        psvd.mesh = &*mesh;
        psvd.mesh_lds_bytes = lds_dwords * 4;
    }
    if (geometry) psvd.geometry = geometry->max_vertices;
    psvd.bindings = bindings;
    psvd.threads = threads.value_or(std::array<std::uint32_t, 3>{});
    psvd.wave_size = wave_size;
    const std::vector<std::uint8_t> psv0 = container::psv(psvd);
    const std::uint8_t hashpart[20] = {};
    const std::vector<std::uint8_t> psg1 = container::signature(prims, false);
    std::vector<container::Part> parts = {
        {{'S', 'F', 'I', '0'}, sfi0},
        {{'I', 'S', 'G', '1'}, isg1},
        {{'O', 'S', 'G', '1'}, osg1},
    };
    if (mesh) parts.push_back({{'P', 'S', 'G', '1'}, psg1});
    parts.push_back({{'P', 'S', 'V', '0'}, psv0});
    parts.push_back({{'H', 'A', 'S', 'H'}, std::span<const std::uint8_t>(hashpart, 20)});
    parts.push_back({{'D', 'X', 'I', 'L'}, dxil});
    return container::build(parts);
}

}  // namespace x2w::vs::shader
