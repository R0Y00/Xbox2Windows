// Instruction inputs, interpolation and export targets. See io.h.
#include "x2w_shader/io.h"

#include <algorithm>
#include <functional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "x2w_shader/front_hash.h"

namespace x2w::vs::io {

namespace {

using gfx10::Operand;
using ir::Exit;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;
using native::HwStage;
using native::PsInput;
using native::VsOutput;

ShaderError unsupported(std::size_t offset, const char* what) { return ShaderError::unsupported(offset, what); }

struct PsVgpr {
    std::uint8_t count;
    std::optional<Barycentric> kind;
};

// The pixel input VGPRs in hardware order: one entry per `SPI_PS_INPUT_ADDR` bit.
const PsVgpr PS_VGPRS[16] = {
    {2, Barycentric::PerspectiveSample},
    {2, Barycentric::PerspectiveCenter},
    {2, Barycentric::PerspectiveCentroid},
    {3, Barycentric::PerspectivePullModel},
    {2, Barycentric::LinearSample},
    {2, Barycentric::LinearCenter},
    {2, Barycentric::LinearCentroid},
    {1, std::nullopt},  // line stipple
    {1, std::nullopt},  // position x
    {1, std::nullopt},  // y
    {1, std::nullopt},  // z
    {1, std::nullopt},  // w
    {1, std::nullopt},  // front face
    {1, std::nullopt},  // ancillary
    {1, std::nullopt},  // sample coverage
    {1, std::nullopt},  // fixed-point position
};

Input ps_system(std::size_t bit)
{
    if (bit == 7) return Input::Simple(Input::Kind::LineStipple);
    if (bit >= 8 && bit <= 11) return Input::Position(static_cast<std::uint8_t>(bit - 8));
    if (bit == 12) return Input::Simple(Input::Kind::FrontFace);
    if (bit == 13) return Input::Simple(Input::Kind::Ancillary);
    if (bit == 14) return Input::Simple(Input::Kind::SampleCoverage);
    return Input::Simple(Input::Kind::FixedPosition);
}

using ValueSiteMap = std::unordered_map<Value, Site, fhash::ValueHash>;
using BaryOf = std::function<std::optional<std::pair<Barycentric, std::uint8_t>>(std::optional<Value>)>;

// An interpolation before its attribute is known: `barycentric` none when i or j is computed.
struct Pair {
    std::optional<Barycentric> barycentric;
    std::vector<Site> first;

    Interp with(Attribute attribute) &&
    {
        if (barycentric) return Interp::Interpolated(attribute, *barycentric, std::move(first));
        return Interp::Weighted(attribute);
    }
};

bool has_site(const Io& io, Site s)
{
    return std::any_of(io.interps.begin(), io.interps.end(), [&](const auto& e) { return e.first == s; });
}

// Records interpolation `pair` at `site`, and its first halves too when they evaluate on their own
// (computed barycentrics).
void push_pair(Io& io, Site site, Pair pair, Attribute attribute)
{
    if (!pair.barycentric) {
        for (const Site f : pair.first)
            if (!has_site(io, f)) io.interps.push_back({f, Interp::Weighted(attribute)});
    }
    io.interps.push_back({site, std::move(pair).with(attribute)});
}

struct HalvesError {
    const char* what;
};

// Finds the first halves of interpolations.
struct Halves {
    const Program* p;
    const ValueSiteMap* defs;
    const ValueSiteMap* phis;

    // The interpolation whose second half reads the first half's result `acc`, the barycentric `j`,
    // and attribute `attr`: every first half must read the same attribute channel. With j and every
    // first half's i loaded as one kind's, the pair interpolates with that kind; otherwise each half
    // evaluates on its own (Interp::Weighted). Throws HalvesError.
    Pair pair(const Operand& attr, std::optional<Value> acc, std::optional<Value> j, const BaryOf& barycentric) const
    {
        std::vector<Site> first;
        if (!acc) throw HalvesError{"an interpolation without its first half"};
        std::vector<Value> seen;
        sites(*acc, seen, first);
        std::optional<Barycentric> kind;
        if (const auto bj = barycentric(j); bj && bj->second == 1) kind = bj->first;
        for (const auto& [b, i0] : first) {
            const Inst& inst = p->blocks[b].insts[i0];
            Operand a;
            std::optional<Value> i;
            const auto* vi = std::get_if<gfx10::Vintrp>(&inst.fields);
            const auto* va = std::get_if<gfx10::Valu>(&inst.fields);
            if (vi && vi->src.is_vector() && inst.name() == "v_interp_p1_f32") {
                a = vi->attr;
                i = inst.read(Reg::V(vi->src.reg));
            } else if (va && inst.name() == "v_interp_p1ll_f16") {
                if (va->src.size() > 1 && va->src[1].is_vector()) {
                    a = va->src[0];
                    i = inst.read(Reg::V(va->src[1].reg));
                } else {
                    throw HalvesError{"an interpolation without its first half"};
                }
            } else {
                throw HalvesError{"an interpolation without its first half"};
            }
            if (!(a == attr)) throw HalvesError{"interpolation halves of different attributes"};
            if (kind) {
                const auto bi = barycentric(i);
                if (!(bi && bi->first == *kind && bi->second == 0)) kind = std::nullopt;
            }
        }
        return Pair{kind, std::move(first)};
    }

    // The instructions `v` comes from, through phis (`seen`: phis on the way; a loop back to one adds
    // nothing).
    void sites(Value v, std::vector<Value>& seen, std::vector<Site>& out) const
    {
        if (const auto it = defs->find(v); it != defs->end()) {
            if (std::find(out.begin(), out.end(), it->second) == out.end()) out.push_back(it->second);
            return;
        }
        const auto it = phis->find(v);
        if (it == phis->end()) throw HalvesError{"an interpolation without its first half"};
        const auto [b, i] = it->second;
        if (std::find(seen.begin(), seen.end(), v) == seen.end()) {
            seen.push_back(v);
            for (const Value a : p->blocks[b].phis[i].args) sites(a, seen, out);
        }
    }
};

}  // namespace

namespace detail {

std::vector<std::pair<std::uint8_t, Input>> ps_vgprs(std::uint32_t addr)
{
    std::vector<std::pair<std::uint8_t, Input>> out;
    std::uint8_t next = 0;
    for (std::size_t bit = 0; bit < 16; ++bit) {
        const auto& [count, kind] = PS_VGPRS[bit];
        if (((addr >> bit) & 1) == 0) continue;
        for (std::uint8_t index = 0; index < count; ++index) {
            const Input input = kind ? Input::Bary(*kind, index) : ps_system(bit);
            out.push_back({static_cast<std::uint8_t>(next + index), input});
        }
        next = static_cast<std::uint8_t>(next + count);
    }
    return out;
}

}  // namespace detail

Io raise(const Program& p, const native::NativeProgram& native)
{
    std::vector<std::pair<Reg, Input>> loaded;
    std::span<const PsInput> ps_inputs;
    std::span<const VsOutput> params;
    const auto* ps = std::get_if<native::PsIo>(&native.io);
    const auto* vs = std::get_if<native::VsIo>(&native.io);
    const auto* cs = std::get_if<native::ComputeIo>(&native.io);
    const auto* ms = std::get_if<native::MeshIo>(&native.io);
    if (ps && native.stage == HwStage::Ps) {
        // A pixel shader exports no parameters; `[]` never reaches `Unread`.
        for (const auto& [r, i] : detail::ps_vgprs(ps->input_addr)) loaded.push_back({Reg::V(r), i});
        ps_inputs = ps->inputs;
    } else if (vs && native.stage == HwStage::Vs) {
        loaded = {
            {Reg::V(0), Input::Simple(Input::Kind::VertexId)},
            {Reg::V(2), Input::Simple(Input::Kind::PrimitiveId)},
            {Reg::V(3), Input::Simple(Input::Kind::InstanceId)},
        };
        for (const auto& i : vs->inputs) {
            for (std::uint8_t channel = 0; channel < 4; ++channel) {
                if (((i.mask >> channel) & 1) == 0) continue;
                loaded.push_back({Reg::V(static_cast<std::uint8_t>(i.vgpr + channel)), Input::Attribute(i.row, channel)});
            }
        }
        params = vs->outputs;
    } else if (cs && native.stage == HwStage::Cs) {
        // Compute: thread ids in v0-v2, group ids and the group info in the SGPRs after the user data.
        const auto [ids, info] = cs->system_sgprs();
        for (std::uint8_t c = 0; c < cs->thread_id_vgprs(); ++c) loaded.push_back({Reg::V(c), Input::ThreadIdInGroup(c)});
        for (std::size_t c = 0; c < ids.size(); ++c)
            if (ids[c]) loaded.push_back({Reg::S(*ids[c]), Input::GroupId(static_cast<std::uint8_t>(c))});
        if (info) loaded.push_back({Reg::S(*info), Input::Simple(Input::Kind::GroupInfo)});
    } else if (ms && native.stage == HwStage::Ms) {
        // The group input, and the phis that may carry it past a join.
        std::vector<Value> group;
        for (const auto& i : p.inputs)
            if (i.first == Reg::V(5)) group.push_back(i.second);
        auto in_group = [&](Value v) { return std::find(group.begin(), group.end(), v) != group.end(); };
        // No lane is active over the EXEC-zero edge of s_cbranch_execz or _execnz, the same case as a
        // partial-EXEC write below.
        auto no_lanes = [&](std::size_t q, std::size_t b) {
            const auto& blk = p.blocks[q];
            if (blk.insts.empty() || !blk.exit.is_branch()) return false;
            const std::string_view last = blk.insts.back().name();
            const std::size_t taken = blk.exit.taken, not_taken = blk.exit.not_taken;
            if (last == "s_cbranch_execz") return taken == b && not_taken != b;
            if (last == "s_cbranch_execnz") return not_taken == b && taken != b;
            return false;
        };
        for (;;) {
            std::optional<Value> found;
            for (std::size_t b = 0; b < p.blocks.size() && !found; ++b) {
                for (const auto& f : p.blocks[b].phis) {
                    const auto& preds = p.blocks[b].preds;
                    bool carried = false;
                    const std::size_t n = std::min(f.args.size(), preds.size());
                    for (std::size_t k = 0; k < n && !carried; ++k)
                        carried = in_group(f.args[k]) && !no_lanes(preds[k], b);
                    if (carried && !in_group(f.value)) {
                        found = f.value;
                        break;
                    }
                }
            }
            if (!found) break;
            group.push_back(*found);
        }
        // The group input holds the same value in every lane only as the first lane reads it. A
        // write to v5 reads its old value too, which only lanes outside EXEC keep.
        // shortcut: writes that may keep the old v5 in active lanes (v_fmac_f32's accumulator, a
        // 16-bit half or d16 load, v_writelane_b32, v_movreld_b32, DPP that skips lanes) and
        // partial-EXEC writes pass; check those too if a mesh program writes v5 that way.
        for (const auto& b : p.blocks) {
            for (const auto& i : b.insts) {
                const auto r = i.read(Reg::V(5));
                if (!r || !in_group(*r) || i.name() == "v_readfirstlane_b32") continue;
                bool per_lane = !i.write(Reg::V(5)).has_value();
                if (!per_lane) {
                    const auto ops = ir::operand_reads(p, i);
                    per_lane = std::find(ops.begin(), ops.end(), Reg::V(5)) != ops.end();
                }
                if (per_lane) throw unsupported(i.machine.offset, "the mesh group input read per lane");
            }
        }
        loaded = {
            {Reg::V(5), Input::Simple(Input::Kind::MeshGroup)},
            {Reg::S(3), Input::Simple(Input::Kind::MergedWaveInfo)},
        };
        params = ms->outputs;
    } else {
        throw unsupported(0, "I/O of this stage");
    }
    std::vector<std::pair<Value, Input>> inputs;
    for (const auto& [reg, v] : p.inputs) {
        const auto it = std::find_if(loaded.begin(), loaded.end(), [&](const auto& l) { return l.first == reg; });
        if (it != loaded.end()) inputs.push_back({v, it->second});
    }
    auto input_of = [&](Value v) -> std::optional<Input> {
        for (const auto& i : inputs)
            if (i.first == v) return i.second;
        return std::nullopt;
    };
    auto attribute = [&](const Inst& inst, const Operand& attr) -> Attribute {
        if (!attr.is_attr()) throw unsupported(inst.machine.offset, "an interpolation operand");
        for (const PsInput& i : ps_inputs)
            if (i.attr == attr.index) return Attribute{i.usage, i.row, attr.chan, attr.high};
        throw unsupported(inst.machine.offset, "an interpolated attribute the stage data does not list");
    };
    // A VGPR value's barycentric kind and index, if it is one as loaded.
    const BaryOf barycentric = [&](std::optional<Value> v) -> std::optional<std::pair<Barycentric, std::uint8_t>> {
        if (!v) return std::nullopt;
        const auto i = input_of(*v);
        if (!i || i->kind != Input::Kind::Barycentric) return std::nullopt;
        return std::pair{i->barycentric, i->index};
    };

    Io io;
    io.inputs = inputs;
    if (cs) io.initialized_sgprs = cs->initialized_sgprs();
    ValueSiteMap defs;
    ValueSiteMap phis;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i)
            for (const auto& w : block.insts[i].writes) defs.insert_or_assign(w.second, Site{b, i});
        for (std::size_t i = 0; i < block.phis.size(); ++i) phis.insert_or_assign(block.phis[i].value, Site{b, i});
    }
    const Halves halves{&p, &defs, &phis};
    auto pair_or_throw = [&](std::size_t at, const Operand& attr, std::optional<Value> acc, std::optional<Value> j) {
        try {
            return halves.pair(attr, acc, j, barycentric);
        } catch (const HalvesError& e) {
            throw unsupported(at, e.what);
        }
    };
    std::vector<std::pair<Site, Attribute>> first_halves;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            const Inst& inst = block.insts[i];
            const Site site{b, i};
            const std::size_t at = inst.machine.offset;
            const std::string_view name = inst.name();
            if (const auto* vi = std::get_if<gfx10::Vintrp>(&inst.fields)) {
                const Attribute attr = attribute(inst, vi->attr);
                if (name == "v_interp_mov_f32" && vi->src.is_const()) {
                    io.interps.push_back({site, Interp::Vertex(attr, vi->src.code)});
                } else if (name == "v_interp_p1_f32") {
                    first_halves.push_back({site, attr});
                } else if (name == "v_interp_p2_f32" && vi->src.is_vector()) {
                    if (!vi->dst.is_vector()) throw unsupported(at, "an interpolation operand");
                    const std::uint8_t d = vi->dst.reg;
                    // The first half's result is this one's old dst.
                    Pair interp = pair_or_throw(at, vi->attr, inst.read(Reg::V(d)), inst.read(Reg::V(vi->src.reg)));
                    push_pair(io, site, std::move(interp), attr);
                } else {
                    throw unsupported(at, "an interpolation form");
                }
            } else if (const auto* v = std::get_if<gfx10::Valu>(&inst.fields); v && name.starts_with("v_interp_")) {
                if (name == "v_interp_p1ll_f16") {
                    first_halves.push_back({site, attribute(inst, v->src.at(0))});
                } else if (name == "v_interp_p2_f16") {
                    const Attribute attr = attribute(inst, v->src.at(0));
                    auto read = [&](std::size_t k) -> std::optional<Value> {
                        if (k < v->src.size() && v->src[k].is_vector()) return inst.read(Reg::V(v->src[k].reg));
                        return std::nullopt;
                    };
                    const auto acc = read(2);
                    const auto j = read(1);
                    Pair interp = pair_or_throw(at, v->src[0], acc, j);
                    push_pair(io, site, std::move(interp), attr);
                } else {
                    throw unsupported(at, "an interpolation form");
                }
            } else if (const auto* e = std::get_if<gfx10::Exp>(&inst.fields)) {
                const std::uint8_t t = e->target;
                Target target;
                if (t <= 7) {
                    target = Target::Color(t);
                } else if (t == 8) {
                    target = Target::Simple(Target::Kind::Depth);
                } else if (t == 9) {
                    target = Target::Simple(Target::Kind::Null);
                } else if (t >= 12 && t <= 16) {
                    target = Target::Position(static_cast<std::uint8_t>(t - 12));
                } else if (t == 20) {
                    target = Target::Simple(Target::Kind::Primitive);
                } else if (t == 32 && params.empty()) {
                    target = Target::Simple(Target::Kind::Unread);
                } else if (t >= 32 && t <= 63) {
                    // Bit 7: a mesh program's per-primitive output.
                    const auto o = std::find_if(params.begin(), params.end(),
                                                [&](const VsOutput& x) { return (x.param & 0x7F) == t - 32; });
                    if (o == params.end()) throw unsupported(at, "a parameter export the stage data does not list");
                    if ((o->param & 0x80) != 0)
                        target = Target::PrimitiveParam(o->usage, o->row);
                    else
                        target = Target::Param(o->usage, o->row);
                } else {
                    throw unsupported(at, "an export target");
                }
                Export x;
                x.site = site;
                x.target = target;
                for (std::size_t c = 0; c < 4; ++c) {
                    const auto& s = e->src[c];
                    if (s && s->is_vector()) x.values[c] = inst.read(Reg::V(s->reg));
                }
                x.compressed = e->compr;
                x.done = e->done;
                io.exports.push_back(x);
            }
        }
    }
    if (first_halves.empty()) return io;
    // A loaded i/j pair normally becomes one host interpolation: its first half need not be
    // evaluated. A first half also read as data (or used without a second half) must keep
    // P0 + i * P10. Follow phi uses too, without forcing ordinary pairs to read per-vertex
    // attributes.
    std::unordered_set<Site, fhash::SiteHash> folded;
    for (const auto& [s, i] : io.interps)
        if (i.kind == Interp::Kind::Interpolated) folded.insert(s);
    std::unordered_set<Value, fhash::ValueHash> used;
    for (std::size_t b = 0; b < p.blocks.size(); ++b)
        for (std::size_t i = 0; i < p.blocks[b].insts.size(); ++i) {
            if (folded.count(Site{b, i})) continue;
            for (const auto& r : p.blocks[b].insts[i].reads) used.insert(r.second);
        }
    for (;;) {
        const std::size_t before = used.size();
        for (const auto& b : p.blocks)
            for (const auto& phi : b.phis)
                if (used.count(phi.value)) used.insert(phi.args.begin(), phi.args.end());
        if (used.size() == before) break;
    }
    for (const auto& [site, attr] : first_halves) {
        const auto& writes = p.blocks[site.first].insts[site.second].writes;
        if (!has_site(io, site) &&
            std::any_of(writes.begin(), writes.end(), [&](const auto& w) { return used.count(w.second) != 0; }))
            io.interps.push_back({site, Interp::Weighted(attr)});
    }
    return io;
}

}  // namespace x2w::vs::io
