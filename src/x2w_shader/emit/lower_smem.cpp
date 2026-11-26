// Lower scalar memory operations, descriptor reads, code constants, and typed buffer formats.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

namespace {

// Inspect resource::Source variants from resource.h.

// Construct a source loaded at the specified site and first dword.
Source make_loaded(Site load, std::uint8_t first) {
    return Source::Loaded(load, first);
}

// Return the first dword if this source was loaded at the given site.
std::optional<std::uint8_t> loaded_at(const Source& l, Site site) {
    if (l.kind == Source::Kind::Loaded && l.load == site) {
        return l.first;
    }
    return std::nullopt;
}

// Test whether a built source refers to code memory.
bool built_in_code(const Source& src) {
    return src.kind == Source::Kind::Built && src.address.param == resource::CODE;
}

// Test for the scalar null register.
bool is_null_scalar(const Operand& op) {
    return op.kind == Operand::Kind::Scalar && op.code == gfx10::NULL_CODE;
}

// Test whether the value is a constant.
bool is_const_v(V v) {
    return v.kind == V::Const;
}

std::optional<std::uint32_t> dword_count(std::string_view rest) {
    if (rest == "") return 1u;
    if (rest == "x2") return 2u;
    if (rest == "x4") return 4u;
    if (rest == "x8") return 8u;
    if (rest == "x16") return 16u;
    return std::nullopt;
}

bool contains(std::string_view s, std::string_view x) { return s.find(x) != std::string_view::npos; }

std::string debug_address(const std::optional<Address>& a) {
    if (!a) return "None";
    std::string s = "Some(Address { param: " + std::to_string(a->param) +
                    ", offset: " + std::to_string(a->offset) + ", dynamic: ";
    s += a->dynamic ? "Some(Value(" + std::to_string(a->dynamic->n) + "))" : std::string("None");
    return s + " })";
}

}  // namespace

void Lower::smem(const Inst& inst, Site site) {
    const auto& sm = std::get<gfx10::Smem>(inst.fields);
    const std::optional<Operand>& data = sm.data;
    const std::int32_t offset0 = sm.offset;
    const Operand& soffset = sm.soffset;
    this->root_copy = std::nullopt;
    std::string_view name = inst.name();
    if (auto it = this->scalar.find(site); it != this->scalar.end() && it->second.param == resource::CODE) {
        Address a = it->second;
        return this->code_smem(inst, a);
    }
    if (auto it = this->scalar.find(site); it != this->scalar.end() && it->second.param == resource::MEMORY) {
        return this->memory_smem(inst);
    }
    constexpr std::string_view prefix = "s_buffer_load_dword";
    if (!name.starts_with(prefix)) {
        return this->descriptor_data(inst, site);
    }
    std::uint32_t count = ok_or(dword_count(name.substr(prefix.size())), "a scalar buffer load");
    if (!data || data->kind != Operand::Kind::Scalar) {
        fail("a scalar buffer load");
    }
    std::uint8_t code = data->code;
    const Source* src = nullptr;
    if (auto it = this->descriptors.find(std::pair{site, false}); it != this->descriptors.end()) {
        src = it->second;
    }
    ok_or(src, "an untracked descriptor");
    const UsedResource* kind = this->used_kind(*src);
    std::size_t c = this->cur;
    auto [base, extra] = this->built_offset(*src);
    // Add the base offset modulo 2^32.
    const std::int32_t offset = static_cast<std::int32_t>(static_cast<std::uint32_t>(offset0) + base);
    V addr = this->s.m.i32(static_cast<std::uint32_t>(offset));
    if (extra) {
        addr = this->s.m.bin(c, Bin::Add, addr, *extra);
    }
    if (!is_null_scalar(soffset)) {
        V so = this->operand(inst, soffset);
        addr = this->s.m.bin(c, Bin::Add, addr, so);
    }
    if (built_in_code(*src)) {
        // A descriptor built around a table in the code.
        for (std::uint32_t k = 0; k < count; ++k) {
            V off = this->s.m.i32(4 * k);
            V a = this->s.m.bin(c, Bin::Add, addr, off);
            V x = this->code_dword(a);
            this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), x, std::nullopt);
        }
        return;
    }
    // A buffer the program picks from the view heap by an index it computes (`ResourceHeap`) has
    // no used-resource entry, and the heap holds a buffer view there, not a constant buffer: read
    // as raw bytes, as a table's SRV is below. Read as a constant buffer (the host's validation:
    // "Descriptor Type in Heap: SRV, Register Type: CBV", heap index 17) the loads gave nothing:
    // the title's meshlet culling reads its batch records, mesh headers and instance transforms
    // this way, so it passed no meshlet and every indirect mesh draw ran empty. The host's table
    // of strides by heap slot tells a buffer view (not 0) from anything else, which is still read
    // as a constant buffer; only the load that fits the slot runs.
    std::optional<V> picked;
    if (kind == nullptr) {
        picked = this->slot_stride(*src);
    }
    if (picked) {
        V stride = *picked;
        V zero = this->s.m.i32(0);
        V is_buffer = this->s.m.cmp(this->cur, pred::NE, stride, zero);
        auto at = this->begin_guarded(is_buffer);
        V h = this->handle(site, *src, false, {11, 0});
        std::vector<std::pair<V, V>> raw;
        bool wide = Lower::wide_loads();
        if (wide) {
            std::vector<V> dwords = this->raw_dwords(h, addr, count);
            for (V x : dwords) raw.emplace_back(x, zero);
        }
        for (std::uint32_t k = 0; k < (wide ? 0 : count); ++k) {
            std::size_t b = this->cur;
            V off = this->s.m.i32(4 * k);
            V a = this->s.m.bin(b, Bin::Add, addr, off);
            V mask = this->s.m.i8(1);
            V u = this->s.m.undef(this->s.t.i32);
            V al = this->s.m.i32(4);
            V r = this->s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                             std::vector<V>{h, a, u, mask, al});
            V x = this->s.m.extract(b, r, 0);
            x = this->as_int(x);
            raw.emplace_back(x, zero);
        }
        std::vector<V> raw_v = this->end_guarded(at, raw);
        V other = this->s.m.cmp(this->cur, pred::EQ, stride, zero);
        auto at2 = this->begin_guarded(other);
        V h2 = this->handle(site, *src, false, {13, 0x1'0000});
        std::vector<std::pair<V, V>> both;
        for (std::size_t k = 0; k < raw_v.size(); ++k) {
            V r = raw_v[k];
            std::size_t b = this->cur;
            V off = this->s.m.i32(4 * static_cast<std::uint32_t>(k));
            V a = this->s.m.bin(b, Bin::Add, addr, off);
            V x = this->cbuffer_dword(h2, a);
            both.emplace_back(x, r);
        }
        std::vector<V> both_v = this->end_guarded(at2, both);
        for (std::size_t k = 0; k < both_v.size(); ++k) {
            this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), both_v[k], std::nullopt);
        }
        return;
    }
    if (kind != nullptr && kind->kind == ResourceType::Srv()) {
        // The size in a structure's header (its second dword), which a copy of it loops over.
        bool size = offset == 4 && count == 1 && !extra && is_null_scalar(soffset) &&
                    emit::builder_bounds();
        bool root = false;
        V h;
        if (auto rm = this->root_memory(*src, addr, std::nullopt, false)) {
            h = rm->first;
            addr = rm->second;
            root = true;
        } else {
            h = this->handle(site, *src, false, {11, 0});
        }
        if (Lower::wide_loads()) {
            std::vector<V> dwords = this->raw_dwords(h, addr, count);
            for (std::size_t k = 0; k < dwords.size(); ++k) {
                this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), dwords[k], std::nullopt);
            }
            return;
        }
        std::size_t c2 = this->cur;
        for (std::uint32_t k = 0; k < count; ++k) {
            V off = this->s.m.i32(4 * k);
            V a = this->s.m.bin(c2, Bin::Add, addr, off);
            V mask = this->s.m.i8(1);
            V u = this->s.m.undef(this->s.t.i32);
            V al = this->s.m.i32(4);
            V r = this->s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                             std::vector<V>{h, a, u, mask, al});
            V x = this->s.m.extract(c2, r, 0);
            x = this->as_int(x);
            if (size && root) {
                V most = this->s.m.i32(1u << 20);
                V zero = this->s.m.i32(0);
                V sane = this->s.m.cmp(c2, pred::IULT, x, most);
                x = this->s.m.select(c2, sane, x, zero);
            }
            this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), x, std::nullopt);
        }
        return;
    }
    V h = this->handle(site, *src, false, {13, 0x1'0000});
    std::optional<std::uint32_t> constant;
    if (is_const_v(addr) && is_null_scalar(soffset) && !extra) {
        constant = static_cast<std::uint32_t>(offset);
    }
    // order: lookups only.
    HashMap<std::uint32_t, V> rows;
    for (std::uint32_t k = 0; k < count; ++k) {
        V x;
        if (constant) {
            std::uint32_t byte = *constant + 4 * k;
            V row;
            if (auto it = rows.find(byte >> 4); it != rows.end()) {
                row = it->second;
            } else {
                V ri = this->s.m.i32(byte >> 4);
                V r = this->s.op(c, "cbufferLoadLegacy.i32", Attr::ReadOnly, this->s.t.cbuf_ret, 59,
                                 std::vector<V>{h, ri});
                rows.insert_or_assign(byte >> 4, r);
                row = r;
            }
            x = this->s.m.extract(c, row, (byte >> 2) & 3);
        } else {
            V off = this->s.m.i32(4 * k);
            V a = this->s.m.bin(c, Bin::Add, addr, off);
            x = this->cbuffer_dword(h, a);
        }
        this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), x, std::nullopt);
    }
}

// A descriptor load (`s_load_dword*` from a table or heap) whose dwords the code also reads as
// numbers: the console compiler computes sizes from descriptor bits (an image's width, height,
// depth, levels; a buffer's record count) instead of querying them. Those fields are rebuilt from
// the host view's dimensions in the GFX10 layout (image dword 1 bits 30-31 and dword 2 bits 0-13
// width - 1, dword 2 bits 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
// shortcut: the base level reads 0 (the host view starts at its own level 0, so sizes the code
// shifts by it come out right) and the other dwords (addresses, formats, tiling) have no host
// value: reading them stops the program.
void Lower::descriptor_data(const Inst& inst, Site site) {
    constexpr Why OTHER = "a descriptor's address or format read as data";
    const auto* sm = std::get_if<gfx10::Smem>(&inst.fields);
    if (!sm || !sm->data || sm->data->kind != Operand::Kind::Scalar) {
        fail("a scalar memory read as data");
    }
    const std::uint8_t code = sm->data->code;
    constexpr std::string_view prefix = "s_load_dword";
    std::string_view iname = inst.name();
    if (!iname.starts_with(prefix)) fail("a scalar memory read as data");
    const std::uint32_t count = ok_or(dword_count(iname.substr(prefix.size())), "a scalar memory read as data");
    // How the program uses the descriptor tells the view's kind: an image or buffer instruction
    // before a scalar buffer load, which reads any buffer.
    const Parts& x = this->x;
    // per resource descriptor (in order) whose leaves load from `of`, its site and the first such leaf's first
    // dword.
    auto uses = [&](Site of) {
        std::vector<std::pair<Site, std::uint8_t>> out;
        for (const auto& d : x.res->descriptors) {
            if (d.role != Role::Resource) continue;
            for (const Source* l : d.source.leaves()) {
                if (auto first = loaded_at(*l, of)) {
                    out.emplace_back(d.site, *first);
                    break;
                }
            }
        }
        return out;
    };
    auto by_scalar_load = [&](const std::pair<Site, std::uint8_t>& u) {
        return std::holds_alternative<gfx10::Smem>(
            x.p->blocks.at(u.first.first).insts.at(u.first.second).fields);
    };
    auto use_of = [&](Site of) -> std::optional<std::pair<Site, std::uint8_t>> {
        auto all = uses(of);
        for (const auto& u : all) {
            if (!by_scalar_load(u)) return u;
        }
        if (!all.empty()) return all.front();
        return std::nullopt;
    };
    // The console compiler loads a descriptor again wherever it needs it, and for a size it reads
    // the dwords of a load that nothing uses as a descriptor. Another load of as many dwords from
    // the same address reads the same descriptor, so what uses that one tells this one's kind. The
    // handle is still made from this load's own operands, which hold here.
    std::optional<Address> address;
    if (auto it = this->scalar.find(site); it != this->scalar.end()) address = it->second;
    auto twin = [&]() -> std::optional<std::pair<Site, std::uint8_t>> {
        if (!address) return std::nullopt;
        const Address a = *address;
        auto same_name = [&](Site s) { return x.p->blocks.at(s.first).insts.at(s.second).name() == inst.name(); };
        for (const auto& [s, b] : x.res->scalar) {
            if (s != site && b == a && same_name(s)) {
                if (auto u = use_of(s)) return u;
            }
        }
        return std::nullopt;
    };
    std::optional<std::pair<Site, std::uint8_t>> use_at = use_of(site);
    if (!use_at) use_at = twin();
    // A view whose kind no use tells, but what is read of it does.
    std::optional<bool> sized;
    if (!use_at) sized = this->sized_view(inst, site, code, count);
    // A descriptor nothing uses as one, whose size the code reads all the same (a texture picked
    // from the heap by an index in a constant buffer, its width and height turned into a texel
    // size): the kind from its dwords and its used-resource entry. Only an image whose layers or
    // depth (dword 4) are not read: those are no 2D texture's; and no buffer `sized_view` did not
    // take (a stride of 4 or less may be a typed view's component count, and the address is
    // `view_address`'s below).
    bool layers = false;
    if (auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + 4)))) layers = this->read_as_number(*v);
    std::optional<std::tuple<std::uint8_t, std::pair<std::uint32_t, std::uint32_t>, std::optional<std::uint32_t>, bool, const UsedResource*>> unused;
    if (!use_at && !sized && count == 8 && !layers) {
        unused = this->unused_descriptor_kind(inst, site, code, count);
    }
    std::uint8_t first = 0;
    std::pair<std::uint32_t, std::uint32_t> props{0, 0};
    std::optional<std::uint32_t> image;
    bool uav = false;
    const UsedResource* used = nullptr;
    if (unused) {
        std::tie(first, props, image, uav, used) = *unused;
    } else {
        std::optional<std::pair<Site, std::uint8_t>> chosen = use_at;
        if (!chosen && sized) chosen = std::pair<Site, std::uint8_t>{site, 0};
        if (!chosen) {
            // A descriptor kept per lane in VGPRs (`Resources::lane_copied`): the code only copies
            // its dwords and compares them, lane against first lane, and every load that reaches
            // those lanes reads one address. Each dword stands in as its place in the table, so
            // the lanes compare equal as they do on the console.
            const auto& lc = this->x.res->lane_copied;
            if (std::find(lc.begin(), lc.end(), site) != lc.end()) {
                if (auto it = this->scalar.find(site); it != this->scalar.end()) {
                    Address a = it->second;
                    for (std::uint32_t k = 0; k < count; ++k) {
                        V xk = this->s.m.i32(a.offset / 4 + k);
                        this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), xk, std::nullopt);
                    }
                    return;
                }
            }
            // A sampler's descriptor: no program reads its dwords as numbers, so where one meets
            // numbers at a join the register is dead on that path (`dead`), as a descriptor's
            // address or format is.
            bool sampler = false;
            for (const auto& d : x.res->descriptors) {
                if (d.role != Role::Sampler) continue;
                for (const Source* l : d.source.leaves()) {
                    if (loaded_at(*l, site)) {
                        sampler = true;
                        break;
                    }
                }
                if (sampler) break;
            }
            if (sampler) {
                for (const auto& w : inst.writes) this->stale.insert(w.second);
                fail("a sampler's descriptor read as data");
            }
            // A buffer view the code only takes the address of, to walk the memory behind it (a
            // ray-tracing structure): the address from the host, the other dwords have no value.
            if (count <= 4) {
                if (auto va = this->view_address(site)) {
                    auto [lo, hi] = *va;
                    for (std::uint32_t k = 0; k < count; ++k) {
                        Reg reg = Reg::S(static_cast<std::uint8_t>(code + k));
                        if (k == 0) {
                            this->write(inst, reg, lo, std::nullopt);
                        } else if (k == 1) {
                            this->write(inst, reg, hi, std::nullopt);
                        } else if (auto v = inst.write(reg)) {
                            this->vals.insert_or_assign(*v, Val::Missing(OTHER));
                            this->stale.insert(*v);
                        }
                    }
                    return;
                }
            }
            if (std::getenv("X2W_SHADERCFG_DESC_WHY") != nullptr) {
                bool copied = std::find(lc.begin(), lc.end(), site) != lc.end();
                std::fprintf(stderr, "DESC-WHY %s %.*s scalar=%s lane_copied=%s\n",
                             hex_alt(inst.machine.offset).c_str(), static_cast<int>(iname.size()),
                             iname.data(), debug_address(address).c_str(), copied ? "true" : "false");
            }
            fail("a descriptor read only as data");
        }
        const Site at = chosen->first;
        first = chosen->second;
        const Source src0 = make_loaded(site, first);
        used = this->used_kind(src0);
        const Inst& ui = this->x.p->blocks.at(at.first).insts.at(at.second);
        std::string_view un = ui.name();
        uav = contains(un, "store") || contains(un, "atomic") ||
              (used != nullptr && used->kind == ResourceType::Uav());
        const std::uint32_t u = uav ? 0x1000 : 0;
        if (sized && *sized) {
            props = {2 | u, 0x409};
            image = 2;
        } else if (sized) {
            props = {11 | u, 0};
            image = std::nullopt;
        } else if (std::holds_alternative<gfx10::Image>(ui.fields) && un.starts_with("image_bvh")) {
            // A ray-tracing structure in a buffer of the host: read raw.
            props = {11, 0};
            image = std::nullopt;
        } else if (const auto* im = std::get_if<gfx10::Image>(&ui.fields)) {
            std::uint32_t k;
            switch (im->dim) {
            case 0: k = 1; break;
            case 1: k = 2; break;
            case 2: k = 4; break;
            case 4: k = 6; break;
            case 5: k = 7; break;
            default: fail("an image dimension (cube or MSAA)");
            }
            props = {k | u, 0x409};
            image = k;
        } else if (const auto* bu = std::get_if<gfx10::Buffer>(&ui.fields)) {
            std::uint32_t stride = used != nullptr ? used->stride : 0;
            bool counted = false;
            if (auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + first + 2)))) {
                counted = this->read_as_number(*v);
            }
            if (un.starts_with("buffer_load_format") || un.starts_with("buffer_store_format")) {
                props = {10 | u, 0x409};
            } else if (!counted && bu->idxen && !bu->format && stride >= 4 && stride % 4 == 0) {
                // Nothing reads the count: the query is cleaned up.
                props = {12 | u, stride};
            } else {
                // Structured ones too: the host's views of them are raw (see `buffer`), which
                // count bytes, and a size asked for as a structured view's is that count on AMD.
                props = {11 | u, 0};
            }
            if (used != nullptr && used->kind == ResourceType::Cbv()) {
                for (const auto& w : inst.writes) this->stale.insert(w.second);
                fail("a constant buffer's descriptor read as data");
            }
            image = std::nullopt;
        } else if (std::holds_alternative<gfx10::Smem>(ui.fields) && used != nullptr &&
                   used->kind == ResourceType::Srv()) {
            // Read by scalar buffer loads alone: the raw view `smem` reads it through.
            props = {11, 0};
            image = std::nullopt;
        } else if (std::holds_alternative<gfx10::Smem>(ui.fields)) {
            // A constant buffer: nothing of its descriptor is a number.
            for (const auto& w : inst.writes) this->stale.insert(w.second);
            fail("a constant buffer's descriptor read as data");
        } else {
            fail("a scalar memory read as data");
        }
    }
    const Source src = make_loaded(site, first);
    V h = this->handle(site, src, false, props);
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    Ty dims = this->s.m.named_struct("dx.types.Dimensions", std::vector<Ty>(4, i32t));
    V mip;
    if (image && !uav) {
        mip = this->s.m.i32(0);
    } else {
        mip = this->s.m.undef(i32t);
    }
    V d = this->s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, std::vector<V>{h, mip});
    auto field = [&](std::uint32_t k) { return this->s.m.extract(c, d, k); };
    std::vector<std::pair<std::uint32_t, V>> words;
    if (image) {
        const std::uint32_t kind = *image;
        V one = this->s.m.i32(1);
        V w = field(0);
        V height, depth;
        switch (kind) {
        case 1:
            height = one;
            depth = one;
            break;
        case 2:
            height = field(1);
            depth = one;
            break;
        case 6:
            height = one;
            depth = field(1);
            break;
        default:
            height = field(1);
            depth = field(2);
            break;
        }
        V levels = uav ? one : field(3);
        auto less = [&](V v) { return this->s.m.bin(c, Bin::Sub, v, one); };
        w = less(w);
        height = less(height);
        depth = less(depth);
        levels = less(levels);
        auto k = [&](std::uint32_t v) { return this->s.m.i32(v); };
        V k30 = k(30);
        V k2 = k(2);
        V k14 = k(14);
        V k16 = k(16);
        V mask = k(0x3FFF);
        V d1 = this->s.m.bin(c, Bin::Shl, w, k30);
        V hi = this->s.m.bin(c, Bin::LShr, w, k2);
        hi = this->s.m.bin(c, Bin::And, hi, mask);
        V ht = this->s.m.bin(c, Bin::Shl, height, k14);
        V d2 = this->s.m.bin(c, Bin::Or, hi, ht);
        static constexpr std::uint32_t TY[8] = {0, 8, 9, 0, 10, 0, 12, 13};
        std::uint32_t ty = TY[kind];  // kind is 1, 2, 4, 6 or 7
        V base = k(0xFAC | ty << 28);
        V last = this->s.m.bin(c, Bin::Shl, levels, k16);
        V d3 = this->s.m.bin(c, Bin::Or, base, last);
        words = {{1, d1}, {2, d2}, {3, d3}, {4, depth}};
    } else {
        // A raw view counts bytes; the descriptor's record count is in elements.
        V n = field(0);
        std::uint32_t stride = used != nullptr ? used->stride : 0;
        bool counted = false;
        if (auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + first + 2)))) {
            counted = this->read_as_number(*v);
        }
        if ((props.first & 0xFF) == 11 && stride > 1) {
            V kk = this->s.m.i32(stride);
            n = this->s.m.bin(c, Bin::UDiv, n, kk);
        } else {
            std::optional<V> slot;
            if ((props.first & 0xFF) == 11 && counted) slot = this->slot_stride(src);
            if (slot) {
                // Picked from the heap: the host's table has the stride.
                Ty i32t2 = this->s.t.i32;
                V one = this->s.m.i32(1);
                V kk = this->s.op(c, "binary.i32", Attr::ReadNone, i32t2, 39, std::vector<V>{*slot, one});
                n = this->s.m.bin(c, Bin::UDiv, n, kk);
            }
        }
        words = {{2, n}};
        // Its address too, where the code computes with it and the host can tell it.
        bool computed = false;
        for (std::uint8_t k = 0; k < 2 && !computed; ++k) {
            if (auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + first + k)))) {
                computed = this->computed_with(*v, 6);
            }
        }
        // (A descriptor nothing uses as one gets it whenever the host can tell it, as a buffer
        // whose address alone is read does.)
        if (first == 0 && (computed || unused)) {
            if (auto va = this->view_address(site)) {
                words.emplace_back(0, va->first);
                words.emplace_back(1, va->second);
            }
        }
    }
    for (std::uint32_t k = 0; k < count; ++k) {
        std::uint32_t rel = k - static_cast<std::uint32_t>(first);  // wrapping_sub
        const std::pair<std::uint32_t, V>* found = nullptr;
        for (const auto& w : words) {
            if (w.first == rel) {
                found = &w;
                break;
            }
        }
        if (found) {
            this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), found->second, std::nullopt);
        } else if (auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + k)))) {
            this->vals.insert_or_assign(*v, Val::Missing(OTHER));
            this->stale.insert(*v);
        }
    }
}

// What descriptor_data makes of a descriptor load that nothing uses as a descriptor but whose
// dwords the code reads: the first dword, the handle properties, the image kind (none for a
// buffer), whether it is an unordered view, and its used-resource entry. An image's eight dwords
// read as a 2D texture's, a buffer's four as a raw view's where the code reads its record count
// (dword 2); none for anything else (a constant buffer, a sampler, a buffer whose address alone
// is read).
// shortcut: an image whose kind nothing tells is taken for a 2D texture (the title's reads are
// texel sizes of 2D textures: puddle masks, shadow masks); a host view of another kind would
// answer its own sizes.
std::optional<std::tuple<std::uint8_t, std::pair<std::uint32_t, std::uint32_t>, std::optional<std::uint32_t>, bool, const UsedResource*>>
Lower::unused_descriptor_kind(const Inst& inst, Site site, std::uint8_t code, std::uint32_t count) const {
    // Kept per lane and compared (`Resources::lane_copied`), or a sampler's: the cases
    // `descriptor_data` handles on their own.
    const auto& lc = this->x.res->lane_copied;
    if (std::find(lc.begin(), lc.end(), site) != lc.end()) return std::nullopt;
    for (const auto& d : this->x.res->descriptors) {
        if (d.role != Role::Sampler) continue;
        for (const Source* l : d.source.leaves()) {
            if (loaded_at(*l, site)) return std::nullopt;
        }
    }
    const UsedResource* used = this->used_kind(make_loaded(site, 0));
    if (used != nullptr && !(used->kind == ResourceType::Srv() || used->kind == ResourceType::Uav())) {
        return std::nullopt;
    }
    auto read = [&](std::uint32_t k) {
        auto v = inst.write(Reg::S(static_cast<std::uint8_t>(code + k)));
        if (!v) return false;
        for (const auto& b : this->x.p->blocks) {
            for (const auto& i : b.insts) {
                for (const auto& r : i.reads) {
                    if (r.second == *v) return true;
                }
            }
        }
        return false;
    };
    const bool uav = used != nullptr && used->kind == ResourceType::Uav();
    const std::uint32_t u = uav ? 0x1000 : 0;
    if (count == 8) {
        // width (dwords 1-2), height (2), levels (3), depth (4)
        bool any = false;
        for (std::uint32_t k = 1; k < 5 && !any; ++k) any = read(k);
        if (any && (used == nullptr || used->dwords == 8)) {
            return std::tuple{std::uint8_t{0}, std::pair<std::uint32_t, std::uint32_t>{2 | u, 0x409},
                              std::optional<std::uint32_t>{2}, uav, used};
        }
    } else if (count == 4) {
        // the record count
        if (read(2) && (used == nullptr || used->dwords == 4)) {
            return std::tuple{std::uint8_t{0}, std::pair<std::uint32_t, std::uint32_t>{11 | u, 0},
                              std::optional<std::uint32_t>{}, uav, used};
        }
    }
    return std::nullopt;
}

// `s_load_dword*` from the program's constant table.
void Lower::code_smem(const Inst& inst, Address a) {
    const auto* sm = std::get_if<gfx10::Smem>(&inst.fields);
    if (!sm || !sm->data || sm->data->kind != Operand::Kind::Scalar) {
        fail("a scalar load");
    }
    const std::uint8_t code = sm->data->code;
    constexpr std::string_view prefix = "s_load_dword";
    std::string_view iname = inst.name();
    if (!iname.starts_with(prefix)) fail("a scalar load from the code");
    const std::uint32_t count = ok_or(dword_count(iname.substr(prefix.size())), "a scalar load from the code");
    std::size_t c = this->cur;
    V addr = this->s.m.i32(a.offset);
    if (a.dynamic) {
        const Value value = *a.dynamic;
        V d = this->i32_of(value);
        // A term held per lane reached the address through `v_readfirstlane_b32`: the first lane's.
        bool per_lane = false;
        if (auto it = this->def_block.find(value); it != this->def_block.end()) {
            for (const auto& i : this->x.p->blocks.at(it->second).insts) {
                for (const auto& w : i.writes) {
                    if (w.second == value && w.first.is_v()) {
                        per_lane = true;
                        break;
                    }
                }
                if (per_lane) break;
            }
        }
        if (per_lane) {
            this->s.flags |= flags::WAVE_OPS;
            Ty i32t = this->s.t.i32;
            V first = this->s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, std::vector<V>{d});
            if (this->s.pixel && emit::helper_own()) {
                V helper = this->helper_lane();
                d = this->s.m.select(c, helper, d, first);
            } else {
                d = first;
            }
        }
        addr = this->s.m.bin(c, Bin::Add, addr, d);
    }
    for (std::uint32_t k = 0; k < count; ++k) {
        V off = this->s.m.i32(4 * k);
        V at = this->s.m.bin(c, Bin::Add, addr, off);
        V xk = this->code_dword(at);
        this->write(inst, Reg::S(static_cast<std::uint8_t>(code + k)), xk, std::nullopt);
    }
}

// The dword at a computed byte address of a constant buffer.
V Lower::cbuffer_dword(V h, V a) {
    std::size_t c = this->cur;
    V four = this->s.m.i32(4);
    V ri = this->s.m.bin(c, Bin::LShr, a, four);
    V r = this->s.op(c, "cbufferLoadLegacy.i32", Attr::ReadOnly, this->s.t.cbuf_ret, 59, std::vector<V>{h, ri});
    V two = this->s.m.i32(2);
    V three = this->s.m.i32(3);
    V comp = this->s.m.bin(c, Bin::LShr, a, two);
    comp = this->s.m.bin(c, Bin::And, comp, three);
    V x = this->s.m.extract(c, r, 0);
    for (std::uint32_t j = 1; j < 4; ++j) {
        V e = this->s.m.extract(c, r, j);
        V jc = this->s.m.i32(j);
        V is = this->s.m.cmp(c, pred::EQ, comp, jc);
        x = this->s.m.select(c, is, e, x);
    }
    return x;
}

// `channels` channels of the element at byte `off` of raw buffer `h`, converted from the GFX10
// buffer format `format` as the hardware's typed loads do (channels past the format's read 0, 0,
// 0, 1).
// shortcut: the descriptor's channel swizzle is taken as the identity (it is for raw views and
// the descriptors code builds), elements of 4 bytes or more as dword-aligned.
std::vector<V> Lower::format_load(V h, V off, std::uint8_t format, std::uint32_t channels) {
    auto [bits, numeric] = ok_or(buffer_format(format), "a buffer format");
    std::size_t c = this->cur;
    std::uint32_t total = 0;
    for (std::uint32_t b : bits) total += b;
    V u = this->s.m.undef(this->s.t.i32);
    V align = this->s.m.i32(4);
    auto load = [&](V at, std::uint32_t n) {
        V mask = this->s.m.i8(static_cast<std::uint8_t>((1u << n) - 1));
        V r = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                         std::vector<V>{h, at, u, mask, align});
        std::vector<V> out;
        for (std::uint32_t k = 0; k < n; ++k) {
            V x = this->s.m.extract(c, r, k);
            out.push_back(this->as_int(x));
        }
        return out;
    };
    std::vector<V> words;
    if (total < 32) {
        // A byte or half inside the dword that holds it.
        V not3 = this->s.m.i32(~3u);
        V three = this->s.m.i32(3);
        V at = this->s.m.bin(c, Bin::And, off, not3);
        V w = load(at, 1).at(0);
        V byte = this->s.m.bin(c, Bin::And, off, three);
        V sh = this->s.m.bin(c, Bin::Shl, byte, three);
        words = {this->s.m.bin(c, Bin::LShr, w, sh)};
    } else {
        words = load(off, total / 32);
    }
    auto float_ = [&](V f) { return this->as_int(f); };
    std::vector<V> out;
    std::uint32_t at = 0;
    for (std::uint32_t w : bits) {
        V word = words.at(at / 32);
        V field;
        if (w == 32) {
            field = word;
        } else {
            V sh = this->s.m.i32(at % 32);
            V t = this->s.m.bin(c, Bin::LShr, word, sh);
            V m = this->s.m.i32((1u << w) - 1);
            field = this->s.m.bin(c, Bin::And, t, m);
        }
        at += w;
        // Sign-extended from the field's top bit.
        auto signed_ = [&]() {
            if (w == 32) return field;
            V k = this->s.m.i32(32 - w);
            V t = this->s.m.bin(c, Bin::Shl, field, k);
            return this->s.m.bin(c, Bin::AShr, t, k);
        };
        Ty f32t = this->s.t.f32;
        // A normalized format's ends are exactly 1.0 and -1.0. The host may divide through a
        // reciprocal and land just below 1.0: RDNA 4 loaded 16_16_UNORM's 65535 as the half
        // 0x3BFF, not 0x3C00.
        V x;
        switch (numeric) {
        case Numeric::Unorm: {
            std::uint64_t top = (std::uint64_t{1} << w) - 1;
            V q;
            if (w <= 16) {
                V qi = this->norm_quotient(field, w);
                q = this->as_float(qi);
            } else {
                V f = this->s.m.cast(c, Cast::UiToFp, field, f32t);
                V max = this->s.m.f32(f32_bits(static_cast<float>(top)));
                q = this->s.m.bin(c, Bin::SDiv, f, max);
            }
            V high = this->s.m.i32(static_cast<std::uint32_t>(top));
            V one = this->s.m.f32(f32_bits(1.0f));
            V at_top = this->s.m.cmp(c, pred::EQ, field, high);
            q = this->s.m.select(c, at_top, one, q);
            x = float_(q);
            break;
        }
        case Numeric::Snorm: {
            V sv = signed_();
            std::uint32_t top = (1u << (w - 1)) - 1;
            V q;
            if (w <= 17) {
                // The magnitude's quotient, signed (codes at and below -top are -1.0 below).
                V zero = this->s.m.i32(0);
                V sign = this->s.m.i32(0x8000'0000u);
                V neg = this->s.m.cmp(c, pred::SLT, sv, zero);
                V minus = this->s.m.bin(c, Bin::Sub, zero, sv);
                V m = this->s.m.select(c, neg, minus, sv);
                V qi = this->norm_quotient(m, w - 1);
                sign = this->s.m.select(c, neg, sign, zero);
                qi = this->s.m.bin(c, Bin::Or, qi, sign);
                q = this->as_float(qi);
            } else {
                V f = this->s.m.cast(c, Cast::SiToFp, sv, f32t);
                V max = this->s.m.f32(f32_bits(static_cast<float>(top)));
                q = this->s.m.bin(c, Bin::SDiv, f, max);
            }
            // The lowest value is -1.0 too, as is the one above it.
            V high = this->s.m.i32(top);
            V one = this->s.m.f32(f32_bits(1.0f));
            V at_top = this->s.m.cmp(c, pred::SGE, sv, high);
            q = this->s.m.select(c, at_top, one, q);
            V low = this->s.m.i32(0u - top);  // wrapping_neg
            V minus_one = this->s.m.f32(f32_bits(-1.0f));
            V at_bottom = this->s.m.cmp(c, pred::SLE, sv, low);
            q = this->s.m.select(c, at_bottom, minus_one, q);
            x = float_(q);
            break;
        }
        case Numeric::Uscaled: {
            V f = this->s.m.cast(c, Cast::UiToFp, field, f32t);
            x = float_(f);
            break;
        }
        case Numeric::Sscaled: {
            V sv = signed_();
            V f = this->s.m.cast(c, Cast::SiToFp, sv, f32t);
            x = float_(f);
            break;
        }
        case Numeric::Uint:
            x = field;
            break;
        case Numeric::Sint:
            x = signed_();
            break;
        case Numeric::Float: {
            if (w == 32) {
                x = field;
                break;
            }
            V hv;
            std::uint32_t k = half_shift(w);
            if (k == 0) {
                hv = field;
            } else {
                V sh = this->s.m.i32(k);
                hv = this->s.m.bin(c, Bin::Shl, field, sh);
            }
            V f = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{hv});
            x = float_(f);
            break;
        }
        }
        out.push_back(x);
    }
    std::uint32_t one = (numeric == Numeric::Uint || numeric == Numeric::Sint) ? 1u : f32_bits(1.0f);
    for (std::uint32_t k = static_cast<std::uint32_t>(out.size()); k < channels; ++k) {
        out.push_back(this->s.m.i32(k == 3 ? one : 0));
    }
    if (out.size() > channels) out.resize(channels);  // truncate
    return out;
}

}  // namespace x2w::vs::lower
