// Inputs, user data, descriptors' entries and title memory. See lower_impl.h and
// GMR/vs_port/lower/PORTING.md for lowering conventions.
#include <algorithm>
#include <bit>
#include <string>
#include <vector>

#include "x2w_shader/emit/lower_impl.h"
#include "x2w_shader/spill.h"

namespace x2w::vs::lower {

namespace {

// ---- small adapters to vs-front's representations (resource.h, io.h) ----

std::optional<std::pair<Site, std::uint8_t>> loaded_of(const Source& src) {
    if (src.kind != Source::Kind::Loaded) return std::nullopt;
    return std::pair<Site, std::uint8_t>{src.load, src.first};
}

const Address* built_address(const Source& src) {
    if (src.kind != Source::Kind::Built) return nullptr;
    return &src.address;
}

Source make_loaded(Site site, std::uint8_t first) {
    return Source::Loaded(site, first);
}

bool is_barycentric_input(const Input& i) { return i.kind == Input::Kind::Barycentric; }

}  // namespace

void Lower::inputs() {
    // A later entry for the same value replaces the earlier one.
    HashMap<Value, Input> io_inputs;
    for (const auto& [w, in] : this->x.io->inputs) io_inputs.insert_or_assign(w, in);
    for (const auto& [reg, v] : this->x.p->inputs) {
        auto found = io_inputs.find(v);
        const Input* input = found == io_inputs.end() ? nullptr : &found->second;
        Val val;
        if (reg.is_s() && this->params.contains(v)) {
            continue;  // loaded on first use
        }
        // Most programs only interpolate with them: loaded on first use as a value (`get`).
        if (reg.is_v() && input && is_barycentric_input(*input)) {
            continue;
        }
        if (reg.is_s()) {
            std::uint8_t n = reg.n;
            if (input) {
                val = this->input(*input);
            } else if (n <= 105 && this->x.io->initialized_sgprs.has_value() &&
                       n >= *this->x.io->initialized_sgprs) {
                // Above the SGPRs the hardware initializes, a register holds what the last wave
                // left, so no program depends on it: a read the paths only seem to allow (a
                // branch past the write, taken only with one past the read) reads zero, as VGPRs
                // do.
                val = Val::I32(this->s.m.i32(0));
            } else if (n >= ir::PARKED) {
                // A scalar parked in a VGPR lane (`ir::unspill`), read where the paths only seem
                // to allow it before any save: the lane of a VGPR nobody wrote, and VGPRs nobody
                // set read as zero.
                val = Val::I32(this->s.m.i32(0));
            } else if (this->pops_collision(v)) {
                // No wave overlaps this one (see `ordered_section`).
                val = Val::I32(this->s.m.i32(0));
            } else {
                val = Val::Missing("a hardware SGPR input");
            }
        } else if (reg.is_scc()) {
            val = Val::Missing("SCC on entry");
        } else {
            if (input) {
                val = this->input(*input);
            } else {
                // Registers no one set read as zero (the validator refuses reads of undef).
                val = Val::I32(this->s.m.i32(0));
            }
        }
        this->vals.insert_or_assign(v, val);
    }
}

// Whether the input SGPR `v` is the POPS collision wave id: a program with an ordered section only
// takes its fields apart (bit 31 says that the wave overlaps another, the others which packer and
// waves).
bool Lower::pops_collision(Value v) const {
    if (this->ordered.empty()) {
        return false;
    }
    bool packer = false;
    for (const auto& block : this->x.p->blocks) {
        for (const auto& p : block.phis) {
            if (std::find(p.args.begin(), p.args.end(), v) != p.args.end()) return false;
        }
        for (const auto& i : block.insts) {
            bool reads = std::any_of(i.reads.begin(), i.reads.end(),
                                     [&](const auto& r) { return r.second == v; });
            if (!reads) {
                continue;
            }
            const auto* sop = std::get_if<gfx10::Sop>(&i.fields);
            if (!sop) {
                return false;
            }
            auto name = i.name();
            const Operand* s1 = sop->src.size() > 1 ? &sop->src[1] : nullptr;
            if (name == "s_bfe_u32" && s1 && s1->kind == Operand::Kind::Literal &&
                s1->literal == 0x0002'001Cu) {
                packer = true;  // bits 28 and 29: the packer
            } else if (name == "s_bfe_u32" || name == "s_lshr_b32") {
            } else {
                return false;
            }
        }
    }
    return packer;
}

Val Lower::input(Input input) {
    std::uint8_t kind = 0;
    std::uint8_t col = 0;
    switch (input.kind) {
    case Input::Kind::VertexId:
        kind = 1;
        col = 0;
        break;
    case Input::Kind::InstanceId:
        kind = 2;
        col = 0;
        break;
    case Input::Kind::Position:
        kind = 3;
        col = input.n;
        break;
    case Input::Kind::Attribute: {
        auto found = this->find_input(input.row, input.channel);
        if (found) return this->load_input(found->first, found->second);
        return Val::Missing("a vertex input the signature lacks");
    }
    case Input::Kind::Barycentric:
        return this->barycentric(input.barycentric, input.index);
    case Input::Kind::PrimitiveId:
        return Val::Missing("the primitive id");
    case Input::Kind::FrontFace:
        // The console code tests it against 0 (`v_cmp_ne_u32`), as the host's value can be.
        kind = 13;
        col = 0;
        break;
    case Input::Kind::ThreadIdInGroup:
        return this->thread_id(input.n);
    case Input::Kind::GroupId:
        return this->compute_id("groupId.i32", 94, input.n);
    case Input::Kind::GroupInfo:
        return Val::Missing("the thread-group info SGPR");
    case Input::Kind::MeshGroup: {
        Val g = this->compute_id("groupId.i32", 94, 0);
        if (!g.is_i32()) std::abort();  // unreachable!()
        V n = this->s.m.i32(this->x.mesh_stride);
        return Val::I32(this->s.m.bin(this->s.entry, Bin::Mul, g.v, n));
    }
    case Input::Kind::MergedWaveInfo: {
        // shortcut: only the wave index (bits 24-27) is set; the console's other bits (lane
        // counts) are read by no program.
        V f = this->mesh_thread();
        std::uint32_t tz = static_cast<std::uint32_t>(std::countr_zero(this->wave()));
        V shift = this->s.m.i32(tz);
        V w = this->s.m.bin(this->s.entry, Bin::LShr, f, shift);
        V k24 = this->s.m.i32(24);
        return Val::I32(this->s.m.bin(this->s.entry, Bin::Shl, w, k24));
    }
    default:
        return Val::Missing("a pixel system input");
    }
    auto max_row = [&]() {
        std::uint32_t best = 0;
        bool any = false;
        for (const auto& e : this->s.inputs) {
            if (!e.row) continue;
            std::uint32_t r = *e.row + 1;
            if (!any || r > best) best = r;
            any = true;
        }
        return any ? best : 0u;
    };
    std::uint32_t free = max_row();
    std::optional<std::size_t> pos;
    for (std::size_t k = 0; k < this->s.inputs.size(); ++k) {
        if (this->s.inputs[k].kind == kind) {
            pos = k;
            break;
        }
    }
    std::size_t id;
    if (pos) {
        id = *pos;
        if (kind != 3) {
            auto& row = this->s.inputs.at(id).row;
            if (!row) row = free;
        }
    } else if (kind >= 3) {
        return Val::Missing("a pixel system input the signature lacks");
    } else {
        // The console signature leaves out vertex and instance ids; PC gives them a row after the
        // others.
        std::uint32_t row = max_row();
        static const char* const NAMES[] = {"", "SV_VertexID", "SV_InstanceID"};
        SigElement e;
        e.name = NAMES[kind];
        e.index = 0;
        e.kind = kind;
        e.comp = 5;
        e.interp = 0;
        e.row = row;
        e.col = 0;
        e.cols = 1;
        e.usage = 0;
        e.half = false;
        this->s.inputs.push_back(std::move(e));
        id = this->s.inputs.size() - 1;
    }
    return this->load_input(id, col);
}

// A component of a compute thread or group id, read in the entry block.
Val Lower::compute_id(std::string_view name, std::uint32_t opcode, std::uint8_t c) {
    std::size_t entry = this->s.entry;
    V k = this->s.m.i32(std::uint32_t{c});
    return Val::I32(
        this->s.op(entry, name, Attr::ReadNone, this->s.t.i32, opcode, std::vector<V>{k}));
}

std::optional<std::pair<std::size_t, std::uint8_t>> Lower::find_input(std::uint8_t row,
                                                                      std::uint8_t channel) const {
    for (std::size_t id = 0; id < this->s.inputs.size(); ++id) {
        const auto& e = this->s.inputs[id];
        if (!e.row) continue;
        std::uint32_t r = *e.row;
        // `then`, not `then_some`: the channel is only past `col` when the condition holds.
        if (r == std::uint32_t{row} && channel >= e.col &&
            channel < static_cast<std::uint8_t>(e.col + e.cols)) {
            return std::pair<std::size_t, std::uint8_t>{id,
                                                        static_cast<std::uint8_t>(channel - e.col)};
        }
    }
    return std::nullopt;
}

Val Lower::load_input(std::size_t id, std::uint8_t col) {
    if (this->s.inputs.at(id).half) {
        // shortcut: 16-bit vertex inputs (the fetch routine's) are not unpacked; pixel inputs
        // are (`interp`).
        return Val::Missing("a 16-bit input");
    }
    return this->load_any_input(id, col);
}

// An input element's channel as declared (16-bit ones as 32-bit floats).
Val Lower::load_any_input(std::size_t id, std::uint8_t col) {
    auto& e = this->s.inputs.at(id);
    e.usage = static_cast<std::uint8_t>(e.usage | (1u << col));
    bool int_ = e.comp != 9;
    bool bit = e.comp == 1;
    Ty ty = bit ? this->s.t.i1 : int_ ? this->s.t.i32 : this->s.t.f32;
    const char* suffix = bit ? ".i1" : int_ ? ".i32" : ".f32";
    std::size_t entry = this->s.entry;
    V a0 = this->s.m.i32(static_cast<std::uint32_t>(id));
    V a1 = this->s.m.i32(0);
    V a2 = this->s.m.i8(col);
    V a3 = this->s.m.undef(this->s.t.i32);
    V v = this->s.op(entry, std::string("loadInput") + suffix, Attr::ReadNone, ty, 4,
                     std::vector<V>{a0, a1, a2, a3});
    if (bit) {
        V z = this->s.m.cast(entry, Cast::ZExt, v, this->s.t.i32);
        return Val::I32(z);
    } else if (int_) {
        return Val::I32(v);
    } else {
        return Val::I32(this->s.m.cast(entry, Cast::Bitcast, v, this->s.t.i32));
    }
}

// The DXIL form of an IR value.
Val Lower::get(Value v) {
    if (this->phi_at.contains(v)) {
        this->read_phis.insert(v);
    }
    bool is_mask = this->x.st->masks.contains(v) || this->x.st->highs.contains(v);
    {
        auto it = this->vals.find(v);
        bool none = it == this->vals.end();
        if ((none || it->second.is_missing()) && this->mask_ok && is_mask) {
            return Val::I32(this->mask_dword(v));
        }
        if (!none) {
            if (it->second.is_missing()) fail(it->second.why);
            return it->second;
        }
    }
    auto pit = this->params.find(v);
    if (pit != this->params.end()) {
        auto [param, dword] = pit->second;
        Val val = this->user(param, dword);
        this->vals.insert_or_assign(v, val);
        return val;
    }
    if (this->lazy_barycentric(v)) {  // Some(Err) throws
        return this->vals.at(v);
    }
    fail("a value defined after its use");
}

// Loads `v` if it is a barycentric input VGPR not loaded yet (see `inputs`): false (None) if it is
// not one; Some(Err) throws.
bool Lower::lazy_barycentric(Value v) {
    const Input* found = nullptr;
    for (const auto& [w, i] : this->x.io->inputs) {
        if (is_barycentric_input(i) && w == v) {
            found = &i;
            break;
        }
    }
    if (!found) return false;
    Barycentric kind = found->barycentric;
    std::uint8_t index = found->index;
    Val val = this->barycentric(kind, index);
    this->vals.insert_or_assign(v, val);
    if (val.is_missing()) fail(val.why);
    return true;
}

// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of `SV_Barycentrics` declared
// with the same interpolation mode.
//
// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are the weights of vertices 1 and
// 2. The host's weights (x, y, z) are those of vertices 0, 1 and 2, vertex 0 being the provoking
// vertex, which is the console's P0 (`v_interp_mov` P0 is lowered to a flat input): i is y and j
// is z.
// shortcut: triangles only; on lines and points the console's i and j are not checked against
// the host's weights.
Val Lower::barycentric(Barycentric kind, std::uint8_t index) {
    auto mode_opt = interp_mode(kind);
    if (!mode_opt) {
        return Val::Missing("pull-model barycentrics");
    }
    std::uint8_t mode = *mode_opt;
    if (index > 1) {
        return Val::Missing("pull-model barycentrics");
    }
    // One element per interpolation mode (the console's own, if its signature declares one):
    // DXIL allows two.
    auto bary = [](const SigElement& e) { return e.kind == 28; };
    std::optional<std::size_t> pos;
    for (std::size_t k = 0; k < this->s.inputs.size(); ++k) {
        const auto& e = this->s.inputs[k];
        if (bary(e) && (e.interp == mode || e.usage == 0)) {
            pos = k;
            break;
        }
    }
    std::size_t id;
    if (pos) {
        id = *pos;
    } else {
        std::uint32_t n = static_cast<std::uint32_t>(
            std::count_if(this->s.inputs.begin(), this->s.inputs.end(), bary));
        if (n >= 2) {
            return Val::Missing("barycentrics of a third interpolation mode");
        }
        // A system value the validator wants unpacked (packing location -1), like other
        // non-packed system values.
        SigElement e;
        e.name = "SV_Barycentrics";
        e.index = n;
        e.kind = 28;
        e.comp = 9;
        e.interp = mode;
        e.row = std::nullopt;
        e.col = 0;
        e.cols = 3;
        e.usage = 0;
        e.half = false;
        this->s.inputs.push_back(std::move(e));
        id = this->s.inputs.size() - 1;
    }
    auto& e = this->s.inputs.at(id);
    if (e.col != 0 || e.cols < 3) {
        return Val::Missing("a console barycentrics element without i and j");
    }
    e.row = std::nullopt;
    e.interp = mode;
    this->s.flags |= flags::BARYCENTRICS;
    return this->load_any_input(id, static_cast<std::uint8_t>(index + 1));
}

V Lower::i32_of(Value v) {
    Val val = this->get(v);
    switch (val.kind) {
    case Val::Kind::I32:
        return val.v;
    case Val::Kind::Bit:
        return this->s.m.cast(this->cur, Cast::ZExt, val.v, this->s.t.i32);
    case Val::Kind::Missing:
    default:
        fail(val.why);
    }
}

// A user-data dword: a root parameter's value.
Val Lower::user(std::size_t param, std::uint32_t dword) {
    const auto rp = this->x.native->layout.at(param);
    if (rp.root == RootType::ResourceHeap() || rp.root == RootType::SamplerHeap()) {
        return Val::I32(this->s.m.i32(0));
    } else if (rp.root == RootType::Constants() || rp.root == RootType::Table()) {
    } else {
        return Val::Missing("a root descriptor");
    }
    std::uint32_t index = std::uint32_t{rp.user_data} + dword - this->user_first();
    UserData u;
    u.index = index;
    u.slot = rp.slot;
    u.root = rp.root;
    u.dword = dword;
    this->user_reads.insert_or_assign(index, u);
    return Val::I32(this->user_at(index));
}

// The user-data index that is dword 0 of the host's user data: 0, or 16 for a program with a
// spilled user-data table, whose parameters all sit at 16 and above. (The host's root signature
// has 64 dwords: a builder stage with constants up to index 45, eight root views, their biases
// and the memory directory would not fit with 16 dwords nothing reads in front.)
std::uint32_t Lower::user_first() const {
    if (spill::is_spilled(*this->x.native)) {
        return 16;
    } else {
        return 0;
    }
}

// A user-data dword no register of the console's holds: one the host computes when it binds (`root` says
// what, an `Other` code), placed after the layout's own dwords.
V Lower::host_word(RootType root, std::uint8_t slot, std::uint32_t dword) {
    std::optional<std::uint32_t> known;
    // Iterate in key order.
    for (const auto& [k, u] : this->user_reads) {
        if (u.root == root && u.slot == slot && u.dword == dword) {
            known = u.index;
            break;
        }
    }
    std::uint32_t index;
    if (known) {
        index = *known;
    } else {
        std::uint32_t first = this->user_first();
        const auto& layout = this->x.native->layout;
        std::uint32_t top = 0;
        for (const auto& e : layout) top = std::max(top, std::uint32_t{e.user_data} + e.sgprs);
        index = top > first ? top - first : 0;  // saturating_sub
        while (this->user_reads.contains(index)) {
            index += 1;
        }
        UserData u;
        u.index = index;
        u.slot = slot;
        u.root = root;
        u.dword = dword;
        this->user_reads.insert_or_assign(index, u);
    }
    return this->user_at(index);
}

// User-data dword `index`, loaded in the entry block.
V Lower::user_at(std::uint32_t index) {
    this->s.user_data = std::max(this->s.user_data, index + 1);
    V h;
    if (this->user_data) {
        h = *this->user_data;
    } else {
        h = this->s.user_data_handle();
        this->user_data = h;
    }
    std::size_t entry = this->s.entry;
    V row = this->s.m.i32(index / 4);
    V r = this->s.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, this->s.t.cbuf_ret, 59,
                     std::vector<V>{h, row});
    return this->s.m.extract(entry, r, index % 4);
}

// Whether some instruction reads `v` as a number (or may, through a join), rather than in a
// descriptor operand.
bool Lower::read_as_number(Value v) const {
    for (const auto& b : this->x.p->blocks) {
        for (const auto& p : b.phis) {
            if (std::find(p.args.begin(), p.args.end(), v) != p.args.end()) return true;
        }
        for (const auto& i : b.insts) {
            bool arith = std::holds_alternative<gfx10::Sop>(i.fields) ||
                         std::holds_alternative<gfx10::Valu>(i.fields);
            if (arith && std::any_of(i.reads.begin(), i.reads.end(),
                                     [&](const auto& r) { return r.second == v; })) {
                return true;
            }
        }
    }
    return false;
}

// Whether some instruction computes with `v` (scalar or vector arithmetic), rather than only naming
// it in a descriptor operand, moving it (at most `depth` moves are followed) or comparing it: a
// waterfall loop keeps a descriptor's dwords per lane, reads the first lane's back and compares
// the lanes' with them.
bool Lower::computed_with(Value v, std::uint32_t depth) const {
    for (const auto& b : this->x.p->blocks) {
        for (const auto& i : b.insts) {
            if (!std::any_of(i.reads.begin(), i.reads.end(),
                             [&](const auto& r) { return r.second == v; })) {
                continue;
            }
            auto name = i.name();
            bool moves = name == "s_mov_b32" || name == "s_mov_b64" || name == "v_mov_b32" ||
                         name == "v_readfirstlane_b32" || name == "v_readlane_b32" ||
                         name == "v_writelane_b32" || name == "v_cndmask_b32";
            bool arith = std::holds_alternative<gfx10::Sop>(i.fields) ||
                         std::holds_alternative<gfx10::Valu>(i.fields);
            bool r;
            if (arith && moves) {
                r = depth > 0 && std::any_of(i.writes.begin(), i.writes.end(), [&](const auto& w) {
                        return this->computed_with(w.second, depth - 1);
                    });
            } else if (arith) {
                r = !(name.starts_with("v_cmp") || name.starts_with("s_cmp"));
            } else {
                r = false;
            }
            if (r) return true;
        }
    }
    return false;
}

// Whether the image `src` names is one of 32-bit integers (one channel): the program has an
// integer atomic on the same table entry, which takes no other view.
bool Lower::is_counter(const Source& src) const {
    auto self_entry = this->entry(src);
    if (!self_entry) {
        return false;
    }
    for (const auto& d : this->x.res->descriptors) {
        const auto& i = this->x.p->blocks.at(d.site.first).insts.at(d.site.second);
        if (d.role != Role::Resource) continue;
        const auto* img = std::get_if<gfx10::Image>(&i.fields);
        if (!img || img->dmask != 1) continue;
        auto n = i.name();
        bool atomic = n == "image_atomic_add" || n == "image_atomic_sub" ||
                      n == "image_atomic_and" || n == "image_atomic_or" ||
                      n == "image_atomic_xor" || n == "image_atomic_smin" ||
                      n == "image_atomic_smax" || n == "image_atomic_umin" ||
                      n == "image_atomic_umax";
        if (!atomic) continue;
        if (this->entry(d.source) == self_entry) return true;
    }
    return false;
}

// Which table entry `src` is loaded from: its root parameter and byte.
std::optional<std::pair<std::size_t, std::uint32_t>> Lower::entry(const Source& src) const {
    auto l = loaded_of(src);
    if (!l) {
        return std::nullopt;
    }
    auto it = this->scalar.find(l->first);
    if (it == this->scalar.end() || it->second.dynamic.has_value()) return std::nullopt;
    const Address& a = it->second;
    return std::pair<std::size_t, std::uint32_t>{a.param, a.offset + 4 * std::uint32_t{l->second}};
}

// See Lower::coherent.
void Lower::find_coherent() {
    std::vector<std::pair<std::size_t, std::uint32_t>> found;
    for (const auto& d : this->x.res->descriptors) {
        const auto& i = this->x.p->blocks.at(d.site.first).insts.at(d.site.second);
        const auto* buf = std::get_if<gfx10::Buffer>(&i.fields);
        if (!(buf && buf->cache.glc)) continue;
        if (!i.name().starts_with("buffer_load_dword")) continue;
        const UsedResource* u = this->used_kind(d.source);
        if (!(u && u->kind == ResourceType::Uav())) continue;
        if (auto e = this->entry(d.source)) found.push_back(*e);
    }
    for (const auto& e : found) this->coherent.insert(e);
}

// The kind of a view no instruction uses (in this program, or through another load of its
// address), by what the code reads of the descriptor the load `inst` at `site` writes to the
// SGPRs from `code`: `Some(true)` for a 2D image, read only for its size, `Some(false)` for a
// structured buffer, read only for its record count.
//
// - Eight dwords are an image's. `GetDimensions` of a 2D one reads the width (dwords 1 and 2), the
//   height (dword 2, bits 14-29, with one `s_bfe_u32`) and the base level or the last (dword 3);
//   of an array, a cube or a volume also dword 4.
// - Four dwords at a table entry the used resources list as an SRV or UAV are a buffer's. A stride
//   above 4 bytes there is an element's (a typed or raw view has at most 4): a test for a null
//   view reads the record count, nothing else.
std::optional<bool> Lower::sized_view(const Inst& inst, Site site, std::uint8_t code,
                                      std::uint32_t count) const {
    auto dword = [&](std::uint8_t k) {
        return inst.write(Reg::S(static_cast<std::uint8_t>(code + k)));
    };
    auto read = [&](Value v) {
        for (const auto& b : this->x.p->blocks) {
            for (const auto& i : b.insts) {
                if (std::any_of(i.reads.begin(), i.reads.end(),
                                [&](const auto& r) { return r.second == v; })) {
                    return true;
                }
            }
            for (const auto& p : b.phis) {
                if (std::find(p.args.begin(), p.args.end(), v) != p.args.end()) return true;
            }
        }
        return false;
    };
    auto only = [&](std::initializer_list<std::uint8_t> allowed) {
        std::uint8_t n = static_cast<std::uint8_t>(count);  // `count as u8`
        for (std::uint8_t k = 0; k < n; ++k) {
            bool ok = std::find(allowed.begin(), allowed.end(), k) != allowed.end();
            if (!ok) {
                auto d = dword(k);
                ok = !(d.has_value() && read(*d));
            }
            if (!ok) return false;
        }
        return true;
    };
    switch (count) {
    case 8: {
        constexpr std::uint32_t HEIGHT = (16u << 16) | 14;
        auto v = dword(2);
        if (!v) return std::nullopt;
        bool height = false;
        for (const auto& b : this->x.p->blocks) {
            for (const auto& i : b.insts) {
                if (i.name() != "s_bfe_u32") continue;
                if (!std::any_of(i.reads.begin(), i.reads.end(),
                                 [&](const auto& r) { return r.second == *v; })) {
                    continue;
                }
                const auto* sop = std::get_if<gfx10::Sop>(&i.fields);
                if (sop && sop->src.size() > 1 && sop->src[1].kind == Operand::Kind::Literal &&
                    sop->src[1].literal == HEIGHT) {
                    height = true;
                    break;
                }
            }
            if (height) break;
        }
        if (height && only({1, 2, 3})) return true;
        return std::nullopt;
    }
    case 4: {
        const UsedResource* used = this->used_kind(make_loaded(site, 0));
        if (!used) return std::nullopt;
        auto it = this->scalar.find(site);
        if (it == this->scalar.end()) return std::nullopt;
        const Address& a = it->second;
        bool buffer = used->dwords == 4 && !a.dynamic.has_value() &&
                      used->table_offset_dwords == a.offset / 4 &&
                      (used->kind == ResourceType::Srv() || used->kind == ResourceType::Uav()) &&
                      used->stride > 4 && used->stride % 4 == 0;
        if (buffer && only({2})) return false;
        return std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

// The title address (low dword, bits from 32 up) of the buffer view whose descriptor the scalar load at
// `site` reads from the start of a table entry, as the host tells it when it binds (USER_VIEW_ADDRESS);
// none for anything else. (The console's descriptor holds more in the second dword, the stride from bit 16
// up; code that takes the address masks those away.)
std::optional<std::pair<V, V>> Lower::view_address(Site site) {
    auto it = this->scalar.find(site);
    if (it == this->scalar.end()) return std::nullopt;
    const Address a = it->second;
    if (a.param >= this->x.native->layout.size()) return std::nullopt;
    const auto rp = this->x.native->layout[a.param];
    if (rp.root != RootType::Table() || a.dynamic.has_value() || a.offset % 32 != 0) {
        return std::nullopt;
    }
    std::uint32_t dword = a.offset / 4;
    // For equal maximum keys, prefer the last entry.
    const UsedResource* used = nullptr;
    for (const auto& u : this->x.native->used) {
        if (u.root == RootType::Table() && u.user_data == rp.user_data &&
            u.table_offset_dwords <= dword) {
            if (!used || u.table_offset_dwords >= used->table_offset_dwords) used = &u;
        }
    }
    if (!used) return std::nullopt;
    if (used->dwords != 4 || !(used->kind == ResourceType::Srv() || used->kind == ResourceType::Uav())) {
        return std::nullopt;
    }
    V lo = this->host_word(RootType::Other(USER_VIEW_ADDRESS), rp.slot, dword);
    V hi = this->host_word(RootType::Other(USER_VIEW_ADDRESS + 1), rp.slot, dword);
    return std::pair<V, V>{lo, hi};
}

// `lo`, `hi` plus the instruction's byte offset `off`, with the carry.
std::pair<V, V> Lower::address_plus(V lo, V hi, std::int32_t off) {
    if (off == 0) {
        return {lo, hi};
    }
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    V k = this->s.m.i32(static_cast<std::uint32_t>(off));
    V sum = this->s.m.bin(c, Bin::Add, lo, k);
    // Up: a carry when the sum wrapped below `lo`. Down: a borrow when it wrapped above.
    V wrapped = this->s.m.cmp(c, off > 0 ? pred::IULT : pred::IUGT, sum, lo);
    V one = this->s.m.cast(c, Cast::ZExt, wrapped, i32t);
    V hi2 = this->s.m.bin(c, off > 0 ? Bin::Add : Bin::Sub, hi, one);
    return {sum, hi2};
}

// Title memory at the 40-bit address `lo`, `hi` (SLOT_MEMORY): the handle of the raw view of the host copy
// holding it and the byte in that view. A page without a copy reads zeros (and drops writes) through the
// null view at the first slot, and its number is left for the host, for lanes that run (`guard`). Goes on
// in another block. With `store`, a page that is not to be written (MEMORY_READ_ONLY) is the null view.
std::pair<V, V> Lower::memory(V lo, V hi, std::optional<V> guard, bool store) {
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    Ty res = this->s.t.res_ret;
    auto k = [](Lower& l, std::uint32_t v) { return l.s.m.i32(v); };
    V k16 = k(*this, 16);
    V k8 = k(*this, 0xFF);
    V low = this->s.m.bin(c, Bin::LShr, lo, k16);
    V high = this->s.m.bin(c, Bin::And, hi, k8);
    high = this->s.m.bin(c, Bin::Shl, high, k16);
    V page = this->s.m.bin(c, Bin::Or, low, high);
    // Inside the directory whatever the address (lanes that do not run compute one too); its last
    // entry is never filled.
    V last = k(*this, MEMORY_PAGES - 1);
    page = this->s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, std::vector<V>{page, last});
    this->s.m.tag(page, Tag::Clamp);
    V k3 = k(*this, 3);
    V at = this->s.m.bin(c, Bin::Shl, page, k3);
    V directory = this->s.root_srv_handle(SLOT_MEMORY);
    V iu = this->s.m.undef(i32t);
    V both = this->s.m.i8(3);
    V align = k(*this, 4);
    V entry = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, res, 139,
                         std::vector<V>{directory, at, iu, both, align});
    V copy0 = this->s.m.extract(c, entry, 0);
    V start0 = this->s.m.extract(c, entry, 1);
    V number = this->as_int(copy0);
    V start = this->as_int(start0);
    V mask = k(*this, ~MEMORY_READ_ONLY);
    V copy = this->s.m.bin(c, Bin::And, number, mask);
    // A page without a copy: its number, at a place by the number. In a branch: the notes lie in
    // the PC's memory, and a store there with every node of every ray would be most of what a
    // trace does.
    V zero = k(*this, 0);
    V missing = this->s.m.cmp(c, pred::EQ, copy, zero);
    if (guard) {
        missing = this->s.m.bin(c, Bin::And, missing, *guard);
    }
    V places = k(*this, MEMORY_FAULTS - 1);
    V faults = this->s.root_uav_handle(SLOT_FAULTS);
    auto noting = this->begin_guarded(missing);
    c = this->cur;
    V place = this->s.m.bin(c, Bin::URem, page, places);
    V one = k(*this, 1);
    place = this->s.m.bin(c, Bin::Add, place, one);
    V k2 = k(*this, 2);
    place = this->s.m.bin(c, Bin::Shl, place, k2);
    Ty void_ = this->s.t.void_;
    V mask1 = this->s.m.i8(1);
    this->s.op(c, "rawBufferStore.i32", Attr::NoUnwind, void_, 140,
               std::vector<V>{faults, place, iu, page, iu, iu, iu, mask1, align});
    this->end_guarded(noting, {});
    c = this->cur;
    this->page_there = this->s.m.cmp(c, pred::NE, copy, zero);
    if (store) {
        V closed = this->s.m.cmp(c, pred::SLT, number, zero);
        copy = this->s.m.select(c, closed, zero, copy);
    }
    V base = this->host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
    V slot = this->s.m.bin(c, Bin::Add, base, copy);
    // The builder's threads read what others of the dispatch wrote (a refit climbs from children
    // to parents): globally coherent.
    std::uint32_t coherent = spill::is_spilled(*this->x.native) ? 0x4000u : 0u;
    V h = this->s.heap_handle(c, slot, false, true, {11u | 0x1000u | coherent, 0u});
    V k64k = k(*this, 0xFFFF);
    V within = this->s.m.bin(c, Bin::And, lo, k64k);
    V byte = this->s.m.bin(c, Bin::Add, start, within);
    return {h, byte};
}

// The title address in the root view `src` is built around, in a program with a spilled user-data table
// (USER_ROOT_ADDRESS).
std::optional<std::pair<V, V>> Lower::root_address(const Source& src) {
    const Address* address = built_address(src);
    if (!address) {
        return std::nullopt;
    }
    if (address->param >= this->x.native->layout.size()) return std::nullopt;
    const auto rp = this->x.native->layout[address->param];
    if (rp.root != RootType::Descriptor() || !spill::is_spilled(*this->x.native)) {
        return std::nullopt;
    }
    const UsedResource* used = nullptr;
    for (const auto& u : this->x.native->used) {
        if (u.user_data == rp.user_data) {
            used = &u;
            break;
        }
    }
    if (!used) return std::nullopt;
    if (!(used->kind == ResourceType::Srv() || used->kind == ResourceType::Uav())) {
        return std::nullopt;
    }
    V lo = this->host_word(RootType::Other(USER_ROOT_ADDRESS), rp.slot, 0);
    V hi = this->host_word(RootType::Other(USER_ROOT_ADDRESS + 1), rp.slot, 0);
    return std::pair<V, V>{lo, hi};
}

// For an access through such a root view: the raw view of the host copy and the byte in it of
// the view's byte `off` (signed: an offset below the address stays below it). Lower::handle gives
// the same view for the rest of the instruction. None for any other source. Goes on in another
// block.
// shortcut: an access of several dwords is taken not to cross the end of a host copy (the
// builder's records lie on their size).
std::optional<std::pair<V, V>> Lower::root_memory(const Source& src, V off, std::optional<V> guard,
                                                  bool store) {
    auto addr = this->root_address(src);
    if (!addr) return std::nullopt;
    auto [lo, hi0] = *addr;
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    V sum = this->s.m.bin(c, Bin::Add, lo, off);
    V wrapped = this->s.m.cmp(c, pred::IULT, sum, lo);
    V carry = this->s.m.cast(c, Cast::ZExt, wrapped, i32t);
    V k31 = this->s.m.i32(31);
    V sign = this->s.m.bin(c, Bin::AShr, off, k31);
    V hi1 = this->s.m.bin(c, Bin::Add, hi0, carry);
    V hi = this->s.m.bin(c, Bin::Add, hi1, sign);
    if (builder_bounds()) {
        // No structure is near 64 MiB (the greatest seen has 14). One whose memory was written
        // over says it has gigabytes, and the builder's copy of it went through all other
        // structures' memory after it and, past 2 GiB, before it. Nothing is there instead.
        V below = this->s.m.i32(16u << 20);
        V all = this->s.m.i32(80u << 20);
        V zero = this->s.m.i32(0);
        V at = this->s.m.bin(c, Bin::Add, off, below);
        V near_ = this->s.m.cmp(c, pred::IULT, at, all);
        sum = this->s.m.select(c, near_, sum, zero);
        hi = this->s.m.select(c, near_, hi, zero);
    }
    auto [h, byte] = this->memory(sum, hi, guard, store);
    this->root_copy = h;
    return std::pair<V, V>{h, byte};
}

// `global_load_*`, `global_store_*` and `global_atomic_*` on title memory by address
// (`resource::MEMORY`): the address is the VGPR pair, each lane its own.
void Lower::memory_flat(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& f = std::get<gfx10::Flat>(inst.fields);  // let Fields::Flat {..} else unreachable
    bool saddr_ok = !f.saddr.has_value() ||
                    (f.saddr->kind == Operand::Kind::Scalar && f.saddr->code == gfx10::NULL_CODE);
    if (f.lds || !saddr_ok) {
        fail("a global address with a scalar base");
    }
    if (!(f.addr.has_value() && f.addr->kind == Operand::Kind::Vector)) {
        fail("a global address not in VGPRs");
    }
    std::uint8_t areg = f.addr->reg;
    auto word = [&](Lower& l, std::uint8_t k) -> V {
        auto v = ok_or(inst.read(Reg::V(static_cast<std::uint8_t>(areg + k))), "a global address");
        return l.i32_of(v);
    };
    V lo0 = word(*this, 0);
    V hi0 = word(*this, 1);
    auto [lo, hi] = this->address_plus(lo0, hi0, std::int32_t{f.offset});
    auto guard = this->guard(b, i);
    Ty i32t = this->s.t.i32;
    Ty res = this->s.t.res_ret;
    Ty void_ = this->s.t.void_;
    V iu = this->s.m.undef(i32t);
    V align = this->s.m.i32(4);
    std::string_view name = inst.name();
    auto count = [&](std::string_view rest) -> std::uint8_t {
        if (rest == "dword") return 1;
        if (rest == "dwordx2") return 2;
        if (rest == "dwordx3") return 3;
        if (rest == "dwordx4") return 4;
        fail(name.data());
    };
    if (name.starts_with("global_load_")) {
        std::uint8_t n = count(name.substr(std::string_view("global_load_").size()));
        if (!(f.dst.has_value() && f.dst->kind == Operand::Kind::Vector)) {
            fail("a global load result");
        }
        std::uint8_t reg = f.dst->reg;
        auto [h, byte] = this->memory(lo, hi, guard, false);
        std::size_t c = this->cur;
        V mask = this->s.m.i8(static_cast<std::uint8_t>((1u << n) - 1));
        V r = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, res, 139,
                         std::vector<V>{h, byte, iu, mask, align});
        for (std::uint8_t k = 0; k < n; ++k) {
            V x = this->s.m.extract(c, r, std::uint32_t{k});
            x = this->as_int(x);
            this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), x, guard);
        }
        return;
    }
    if (name.starts_with("global_store_")) {
        std::uint8_t n = count(name.substr(std::string_view("global_store_").size()));
        if (!f.data) fail("global store data");
        const Operand& d = *f.data;
        std::vector<V> vals = this->vgprs(inst, d, n);
        auto [h, byte] = this->memory(lo, hi, guard, true);
        std::vector<V> args{h, byte, iu};
        args.insert(args.end(), vals.begin(), vals.end());
        args.resize(3 + 4, iu);
        args.push_back(this->s.m.i8(static_cast<std::uint8_t>((1u << n) - 1)));
        args.push_back(align);
        auto at = this->begin_guarded(guard);
        this->s.op(this->cur, "rawBufferStore.i32", Attr::NoUnwind, void_, 140, args);
        this->end_guarded(at, {});
        return;
    }
    if (!name.starts_with("global_atomic_")) {
        fail(name.data());
    }
    std::string_view op = name.substr(std::string_view("global_atomic_").size());
    if (!f.data) fail("global atomic data");
    const Operand& d = *f.data;
    V v = this->vgprs(inst, d, 1).at(0);
    // With `glc` the value from before comes back, in the result register the instruction names.
    std::optional<std::uint8_t> back;
    if (f.dst.has_value() && f.dst->kind == Operand::Kind::Vector && f.cache.glc) {
        back = f.dst->reg;
    }
    std::optional<Value> back_v;
    if (back) back_v = inst.read(Reg::V(*back));
    V old = back_v ? this->i32_of(*back_v) : this->s.m.i32(0);
    auto [h, byte] = this->memory(lo, hi, guard, true);
    std::size_t c = this->cur;
    // DXIL's atomic operations: add 0, and 1, or 2, xor 3, signed minimum 4 and maximum 5,
    // unsigned 6 and 7, exchange 8.
    std::optional<std::pair<std::uint32_t, V>> simple;
    if (op == "add") {
        simple = std::pair<std::uint32_t, V>{0, v};
    } else if (op == "sub") {
        V zero = this->s.m.i32(0);
        simple = std::pair<std::uint32_t, V>{0, this->s.m.bin(c, Bin::Sub, zero, v)};
    } else if (op == "and") {
        simple = std::pair<std::uint32_t, V>{1, v};
    } else if (op == "or") {
        simple = std::pair<std::uint32_t, V>{2, v};
    } else if (op == "xor") {
        simple = std::pair<std::uint32_t, V>{3, v};
    } else if (op == "smin") {
        simple = std::pair<std::uint32_t, V>{4, v};
    } else if (op == "smax") {
        simple = std::pair<std::uint32_t, V>{5, v};
    } else if (op == "umin") {
        simple = std::pair<std::uint32_t, V>{6, v};
    } else if (op == "umax") {
        simple = std::pair<std::uint32_t, V>{7, v};
    } else if (op == "swap") {
        simple = std::pair<std::uint32_t, V>{8, v};
    } else if (op == "fmin" || op == "fmax") {
        simple = std::nullopt;
    } else {
        fail(name.data());
    }
    auto at = this->begin_guarded(guard);
    V was;
    if (simple) {
        V code = this->s.m.i32(simple->first);
        was = this->s.op(this->cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78,
                         std::vector<V>{h, code, byte, iu, iu, simple->second});
    } else {
        // A float minimum or maximum has no atomic operation on the host: the dword is exchanged
        // for the smaller (larger) of itself and the value until no other thread got in between.
        // Each failed pass means another one's succeeded; the limit only keeps a broken card from
        // hanging.
        std::size_t head = this->cur;
        V mask = this->s.m.i8(1);
        V first = this->s.op(head, "rawBufferLoad.f32", Attr::ReadOnly, res, 139,
                             std::vector<V>{h, byte, iu, mask, align});
        first = this->s.m.extract(head, first, 0);
        first = this->as_int(first);
        std::size_t pass = this->s.m.new_block();
        std::size_t done = this->s.m.new_block();
        this->s.m.terminate(head, Term::br(pass));
        this->cur = pass;
        V seen = this->s.m.phi(pass, i32t);
        V n = this->s.m.phi(pass, i32t);
        Ty f32t = this->s.t.f32;
        V fv = this->as_float(v);
        V fseen = this->as_float(seen);
        // `dx.op.binary`: FMax 35, FMin 36.
        std::uint32_t pick = op == "fmin" ? 36 : 35;
        V wanted = this->s.op(pass, "binary.f32", Attr::ReadNone, f32t, pick,
                              std::vector<V>{fseen, fv});
        wanted = this->as_int(wanted);
        V found = this->s.op(pass, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79,
                             std::vector<V>{h, byte, iu, iu, seen, wanted});
        V one = this->s.m.i32(1);
        V n1 = this->s.m.bin(pass, Bin::Add, n, one);
        V lost = this->s.m.cmp(pass, pred::NE, found, seen);
        V limit = this->s.m.i32(1u << 16);
        V more = this->s.m.cmp(pass, pred::IULT, n1, limit);
        V again = this->s.m.bin(pass, Bin::And, lost, more);
        this->s.m.terminate(pass, Term::cond_br(again, pass, done));
        V zero = this->s.m.i32(0);
        this->s.m.set_phi(seen, std::vector<std::pair<V, std::size_t>>{{first, head}, {found, pass}});
        this->s.m.set_phi(n, std::vector<std::pair<V, std::size_t>>{{zero, head}, {n1, pass}});
        this->cur = done;
        was = found;
    }
    std::vector<std::pair<V, V>> keep{{was, old}};
    was = this->end_guarded(at, keep).at(0);
    if (back) {
        this->write(inst, Reg::V(*back), was, std::nullopt);
    }
}

// `s_load_dword*` from title memory by address (`resource::MEMORY`): the SGPR pair plus the
// instruction's offsets.
void Lower::memory_smem(const Inst& inst) {
    const auto* sm = std::get_if<gfx10::Smem>(&inst.fields);
    if (!(sm && sm->data.has_value() && sm->data->kind == Operand::Kind::Scalar &&
          sm->base.kind == Operand::Kind::Scalar)) {
        fail("a scalar load from memory");
    }
    std::uint8_t code = sm->data->code;
    std::uint8_t base = sm->base.code;
    std::int32_t offset = sm->offset;
    const Operand soffset = sm->soffset;
    std::uint8_t count;
    std::string_view name = inst.name();
    constexpr std::string_view PRE = "s_load_dword";
    if (!name.starts_with(PRE)) fail("a scalar memory access by address");
    std::string_view rest = name.substr(PRE.size());
    if (rest == "") {
        count = 1;
    } else if (rest == "x2") {
        count = 2;
    } else if (rest == "x4") {
        count = 4;
    } else if (rest == "x8") {
        count = 8;
    } else if (rest == "x16") {
        count = 16;
    } else {
        fail("a scalar memory access by address");
    }
    auto word = [&](Lower& l, std::uint8_t k) -> V {
        auto v = ok_or(inst.read(Reg::S(static_cast<std::uint8_t>(base + k))),
                       "an address not in SGPRs");
        return l.i32_of(v);
    };
    V lo = word(*this, 0);
    V hi = word(*this, 1);
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    Ty res = this->s.t.res_ret;
    if (!(soffset.kind == Operand::Kind::Scalar && soffset.code == gfx10::NULL_CODE)) {
        // An offset register counts up from the address.
        V so = this->operand(inst, soffset);
        V sum = this->s.m.bin(c, Bin::Add, lo, so);
        V carry = this->s.m.cmp(c, pred::IULT, sum, lo);
        carry = this->s.m.cast(c, Cast::ZExt, carry, i32t);
        hi = this->s.m.bin(c, Bin::Add, hi, carry);
        lo = sum;
    }
    V iu = this->s.m.undef(i32t);
    V align = this->s.m.i32(4);
    for (std::uint8_t first = 0; first < count; first += 4) {
        std::uint8_t n = std::min<std::uint8_t>(static_cast<std::uint8_t>(count - first), 4);
        auto [alo, ahi] = this->address_plus(lo, hi, offset + 4 * std::int32_t{first});
        auto [h, byte] = this->memory(alo, ahi, std::nullopt, false);
        std::size_t c2 = this->cur;
        V mask = this->s.m.i8(static_cast<std::uint8_t>((1u << n) - 1));
        V r = this->s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, res, 139,
                         std::vector<V>{h, byte, iu, mask, align});
        for (std::uint8_t k = 0; k < n; ++k) {
            V x = this->s.m.extract(c2, r, std::uint32_t{k});
            x = this->as_int(x);
            this->write(inst, Reg::S(static_cast<std::uint8_t>(code + first + k)), x, std::nullopt);
        }
    }
}

// A dword of a lane mask as a number: the ballot of the lanes' bits, taken in the block that
// writes the mask (see MaskData).
V Lower::mask_dword(Value v) {
    Value lo = v;
    bool high = false;
    if (auto it = this->x.st->highs.find(v); it != this->x.st->highs.end()) {
        lo = it->second;
        high = true;
    }
    auto mit = this->x.st->masks.find(lo);
    if (mit == this->x.st->masks.end()) fail(LANE_MASK);
    Bool m = mit->second;
    // A mask that holds lanes outside EXEC where it is written (the complement of one, say) is the
    // ballot of the lanes that are there: all of them, where two exchanging host waves know every
    // lane runs that block.
    std::optional<std::size_t> made;
    if (auto it = this->def_block.find(lo); it != this->def_block.end()) made = it->second;
    bool whole = this->exchange.has_value() && !this->local && (!made || this->all_lanes(*made));
    if (!this->x.st->active.contains(lo) && !whole) {
        if (this->exchange.has_value() && !this->local) {
            need(made.value_or(0));
        }
        fail(MASK_ELSEWHERE);
    }
    std::pair<V, V> xy;
    if (auto it = this->ballots.find(lo); it != this->ballots.end()) {
        xy = it->second;
    } else {
        std::size_t at;
        auto db = this->def_block.find(lo);
        if (db == this->def_block.end()) {
            at = this->s.entry;  // Program inputs (EXEC on entry).
        } else if (db->second == this->cur_ir) {
            at = this->cur;
        } else {
            at = ok_or(this->tail.at(db->second), "a lane mask written after its use");
        }
        if (!this->balloting.insert(lo).second) {
            fail("a lane mask whose lanes depend on its own value");
        }
        V bit;
        try {
            bit = this->boolean(m, at);
        } catch (const WhyFail&) {
            this->balloting.erase(lo);
            throw;
        }
        this->balloting.erase(lo);
        Ty i32t = this->s.t.i32;
        Ty four = this->s.m.named_struct("dx.types.fouri32", std::vector<Ty>(4, i32t));
        V r = this->s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, std::vector<V>{bit});
        this->s.flags |= flags::WAVE_OPS;
        V x0 = this->s.m.extract(at, r, 0);
        V y0 = this->s.m.extract(at, r, 1);
        xy = {x0, y0};
        if (this->exchange.has_value()) {
            // a host wave's ballot is its half's 32 bits: the halves exchange them
            std::optional<std::size_t> b = made;  // def_block.get(&lo), as above
            if (this->local) {
                if (!this->local_ok && !this->local_any) {
                    fail("a lane mask read as data in a half-wave");
                }
            } else if (!std::get<1>(*this->exchange) && !(!b || this->all_lanes(*b))) {
                // (a group of 32 or fewer has no other half to exchange with)
                need(b.value_or(0));
                fail("a lane mask exchanged where not every lane runs");
            }
            xy = this->exchange_halves(at, xy.first);
        }
        this->ballots.insert_or_assign(lo, xy);
    }
    return high ? xy.second : xy.first;
}

}  // namespace x2w::vs::lower
