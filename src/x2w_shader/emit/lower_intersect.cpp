// The console's ray-tracing node test, `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`, as inline
// DXIL: the host has no instruction that tests one ray against one node of the title's own structures.
//
// bvh.h (crate::bvh) says what the instruction computes and is the reference: the code made here
// does the same float operations in the same order (DXIL float operations carry no fast-math
// flags, so the host may not reorder or fuse them), and the tests in intersect_tests run it on the
// card against the reference.
//
// The title's nodes are in title memory, named by address: they are read through the host's
// copies of title memory (Lower::memory), one look in its directory per node (two for a 128-byte
// box node).
//
// Shape of the code, under the instruction's lane guard:
//
//   kind = pointer & 7
//   box (4, 5):  4 child pointers; 24 box floats (kind 5) or 24 halves widened (kind 4);
//                four slab tests; the sorting network
//   triangle (0-3): the id dword, three vertices at the bytes it names; the watertight test
//   other (6, 7): a triangle miss
//
// Each lane reads only the dwords its node's kind needs: 28 for a 32-bit box node, 16 for a 16-bit
// one, 10 for a triangle.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "x2w_shader/bvh.h"
#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

// The instruction's node resides in a buffer or in mapped memory.
struct NodeAt {
    enum class Kind : std::uint8_t {
        // At this byte of a raw buffer of the host.
        Buffer,
        // In title memory: the raw view of the host copy holding the node's first 64 bytes and the
        // node's byte in it (Lower::bvh_memory), the node's address, for the second half of a
        // 128-byte node, and whether there is a copy.
        Memory,
    };
    Kind kind = Kind::Buffer;
    V handle{};
    V byte{};
    V low{};     // Memory
    V high{};    // Memory
    V mapped{};  // Memory

    static NodeAt Buffer(V handle, V byte) {
        NodeAt n;
        n.kind = Kind::Buffer;
        n.handle = handle;
        n.byte = byte;
        return n;
    }
    static NodeAt Memory(V handle, V byte, V low, V high, V mapped) {
        NodeAt n;
        n.kind = Kind::Memory;
        n.handle = handle;
        n.byte = byte;
        n.low = low;
        n.high = high;
        n.mapped = mapped;
        return n;
    }
};

// The ray operands, as floats.
struct Ray {
    V extent{};
    std::array<V, 3> origin{};
    std::array<V, 3> direction{};
    std::array<V, 3> inverse{};
};

// `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`.
void Lower::bvh(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& img = std::get<gfx10::Image>(inst.fields);
    const Operand& data = img.data;
    const auto& addr = img.addr;
    const Operand& rsrc = img.rsrc;
    if (img.a16 || img.d16 || img.tfe || img.lwe || img.dmask != 0xf) {
        fail("a ray-tracing node test with 16-bit operands or a partial result");
    }
    const bool wide = inst.name() == "image_bvh64_intersect_ray";
    // Operand VGPRs in order: the node pointer (two dwords in the 64-bit form), the ray's extent,
    // origin, direction and inverse direction.
    std::vector<std::uint8_t> regs;
    for (const Operand& a : addr) {
        if (a.kind == Operand::Kind::Vector) {
            for (std::uint8_t k = 0; k < a.dwords; ++k) {
                regs.push_back(static_cast<std::uint8_t>(a.reg + k));
            }
        }
    }
    if (regs.size() != (wide ? 12u : 11u)) {
        fail("a ray-tracing node test's operands");
    }
    if (data.kind != Operand::Kind::Vector) {
        fail("a ray-tracing node test's result");
    }
    const std::uint8_t dst = data.reg;
    const Site site{b, i};
    // x.res->bvh: vector<pair<Site, BvhNodes>>.
    std::optional<resource::BvhNodes> found;
    for (const auto& n : this->x.res->bvh) {
        if (n.first == site) {
            found = n.second;
            break;
        }
    }
    const resource::BvhNodes nodes = ok_or(found, "an untracked ray-tracing structure");
    bool sort = false;
    bool barycentrics = false;
    if (nodes.kind == resource::BvhNodes::Kind::Resource) {
        sort = true;
        barycentrics = true;
    } else {
        sort = (nodes.word1 & bvh::BOX_SORT) != 0;
        barycentrics = (nodes.word3 & bvh::RETURN_BARYCENTRICS) != 0;
    }
    if (!barycentrics) {
        // Mode 0 returns a triangle id and a hit status in the last two dwords; no program of the
        // title asks for it.
        fail("a ray-tracing structure that returns triangle ids");
    }
    const std::optional<V> guard = this->guard(b, i);
    std::vector<V> x;
    x.reserve(regs.size());
    for (std::uint8_t r : regs) {
        const Value v = ok_or(inst.read(Reg::V(r)), "an unread operand");
        x.push_back(this->i32_of(v));
    }
    const V pointer = x[0];
    const std::optional<V> pointer_high = wide ? std::optional<V>(x[1]) : std::nullopt;
    std::vector<V> f;
    for (std::size_t n = 1 + (wide ? 1 : 0); n < x.size(); ++n) {
        f.push_back(this->as_float(x[n]));
    }
    Ray ray;
    ray.extent = f.at(0);
    ray.origin = {f.at(1), f.at(2), f.at(3)};
    ray.direction = {f.at(4), f.at(5), f.at(6)};
    ray.inverse = {f.at(7), f.at(8), f.at(9)};
    const std::size_t c = this->cur;
    NodeAt at;
    if (nodes.kind == resource::BvhNodes::Kind::Resource) {
        auto d = this->descriptors.find(std::pair<Site, bool>{site, false});
        if (d == this->descriptors.end()) fail("an untracked ray-tracing structure");
        const Source* src = d->second;
        const V handle = this->handle(site, *src, false, {11, 0});
        // (pointer >> 3) << 6 from the buffer's start.
        const V kind_bits = this->s.m.i32(~7u);
        const V k3 = this->s.m.i32(3);
        V byte = this->s.m.bin(c, Bin::And, pointer, kind_bits);
        byte = this->s.m.bin(c, Bin::Shl, byte, k3);
        at = NodeAt::Buffer(handle, byte);
    } else {
        if (rsrc.kind != Operand::Kind::Scalar) {
            fail("a ray-tracing structure descriptor not in SGPRs");
        }
        const std::uint8_t code = rsrc.code;
        std::array<V, 2> words{pointer, pointer};
        for (std::size_t k = 0; k < 2; ++k) {
            const Value v = ok_or(inst.read(Reg::S(static_cast<std::uint8_t>(code + k))), "an unread operand");
            words[k] = this->i32_of(v);
        }
        const auto [low, high] = this->bvh_address(words[0], words[1], pointer, pointer_high);
        // One look for the node's copy, before the branches: a node is 64 bytes on 64, so its
        // first half lies in one 64 KiB page whatever its kind.
        const auto [handle, byte, mapped] = this->bvh_memory(low, high, guard);
        at = NodeAt::Memory(handle, byte, low, high, mapped);
    }
    const V zero = this->s.m.i32(0);
    const auto gate = this->begin_guarded(guard);
    const std::size_t head = this->cur;
    const V k4 = this->s.m.i32(4);
    const V k6 = this->s.m.i32(6);
    const V k7 = this->s.m.i32(7);
    const V kind = this->s.m.bin(head, Bin::And, pointer, k7);
    const V pair = this->s.m.bin(head, Bin::And, pointer, k6);
    const V is_box = this->s.m.cmp(head, pred::EQ, pair, k4);
    const V is_triangle = this->s.m.cmp(head, pred::IULT, kind, k4);
    const std::size_t box_block = this->s.m.new_block();
    const std::size_t leaf_block = this->s.m.new_block();
    const std::size_t triangle_block = this->s.m.new_block();
    const std::size_t join = this->s.m.new_block();
    this->s.m.terminate(head, Term::cond_br(is_box, box_block, leaf_block));
    this->s.m.terminate(leaf_block, Term::cond_br(is_triangle, triangle_block, join));
    this->cur = box_block;
    const std::array<V, 4> boxes = this->bvh_boxes(at, kind, ray, sort);
    const std::size_t box_end = this->cur;
    this->s.m.terminate(box_end, Term::br(join));
    this->cur = triangle_block;
    const std::array<V, 4> triangle = this->bvh_triangle(at, kind, ray);
    const std::size_t triangle_end = this->cur;
    this->s.m.terminate(triangle_end, Term::br(join));
    this->cur = join;
    // Kinds 6 and 7 (an instance, a procedural primitive): not known what the console returns;
    // the title's programs do not read it.
    std::array<V, 4> other{};
    for (std::size_t k = 0; k < 4; ++k) {
        other[k] = this->s.m.i32(f32_bits(bvh::MISS[k]));
    }
    const Ty i32t = this->s.t.i32;
    std::vector<std::pair<V, V>> results;
    results.reserve(4);
    for (std::size_t k = 0; k < 4; ++k) {
        const V phi = this->s.m.phi(join, i32t);
        this->s.m.set_phi(phi, std::vector<std::pair<V, std::size_t>>{
                                   {boxes[k], box_end},
                                   {triangle[k], triangle_end},
                                   {other[k], leaf_block},
                               });
        results.emplace_back(phi, zero);
    }
    const std::vector<V> out = this->end_guarded(gate, results);
    for (std::size_t k = 0; k < out.size(); ++k) {
        this->write(inst, Reg::V(static_cast<std::uint8_t>(dst + k)), out[k], guard);
    }
}

// Title memory at `low`/`high` for a node's bytes: the raw view of the host copy holding the address and
// the byte in that view (Lower::memory: a page the host has no copy of reads zeros and is left for the host
// to fetch, for lanes that run).
std::tuple<V, V, V> Lower::bvh_memory(V low, V high, std::optional<V> guard) {
    // Device tests can override test_memory().
    if (const auto tm = test_memory()) {
        const auto [slot_n, base] = *tm;
        // The buffer at heap slot `slot` holds the bytes from title address `base` on; any other
        // address reads past its end, which is zeros.
        const std::size_t c = this->cur;
        const V slot = this->s.m.i32(slot_n);
        const V handle = this->s.heap_handle(c, slot, false, false, {11, 0});
        const V base_low = this->s.m.i32(static_cast<std::uint32_t>(base));
        const V base_high = this->s.m.i32(static_cast<std::uint32_t>(base >> 32));
        V at = this->s.m.bin(c, Bin::Sub, low, base_low);
        const V in_low = this->s.m.cmp(c, pred::IUGE, low, base_low);
        const V in_high = this->s.m.cmp(c, pred::EQ, high, base_high);
        const V inside = this->s.m.bin(c, Bin::And, in_low, in_high);
        const V outside = this->s.m.i32(0xffff'ff00u);
        at = this->s.m.select(c, inside, at, outside);
        return {handle, at, inside};
    }
    const auto [handle, byte] = this->memory(low, high, guard, false);
    if (!this->page_there) std::abort();
    const V mapped = *this->page_there;
    return {handle, byte, mapped};
}

// The second 64 bytes of a 128-byte node: in a page, and in a copy, of their own maybe. Goes on in another
// block.
NodeAt Lower::bvh_second_half(NodeAt at) {
    if (at.kind == NodeAt::Kind::Buffer) {
        const V size = this->s.m.i32(64);
        const V byte = this->s.m.bin(this->cur, Bin::Add, at.byte, size);
        return NodeAt::Buffer(at.handle, byte);
    }
    const auto [low, high] = this->address_plus(at.low, at.high, 64);
    // Every lane in this block runs the instruction.
    const auto [handle, byte, mapped] = this->bvh_memory(low, high, std::nullopt);
    return NodeAt::Memory(handle, byte, low, high, mapped);
}

// `count` dwords of 64 bytes of a node from their byte `base` + `offset` (`base` computed, `offset` fixed),
// read four at a time.
std::vector<V> Lower::bvh_node_words(NodeAt at, std::optional<V> base, std::uint32_t offset,
                                     std::uint32_t count) {
    const V undef = this->s.m.undef(this->s.t.i32);
    const V align = this->s.m.i32(4);
    const V handle = at.handle;
    const V start_byte = at.byte;
    std::vector<V> out;
    out.reserve(count);
    for (std::uint32_t start = 0; start < count; start += 4) {
        const std::uint32_t n = std::min<std::uint32_t>(count - start, 4);
        const std::uint32_t node_byte = offset + 4 * start;
        V byte = start_byte;
        const std::size_t c = this->cur;
        if (base) {
            byte = this->s.m.bin(c, Bin::Add, byte, *base);
        }
        const V k = this->s.m.i32(node_byte);
        byte = this->s.m.bin(c, Bin::Add, byte, k);
        const V mask = this->s.m.i8(static_cast<std::uint8_t>((1u << n) - 1));
        const V r = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                               std::vector<V>{handle, byte, undef, mask, align});
        for (std::uint32_t kk = 0; kk < n; ++kk) {
            const V fv = this->s.m.extract(c, r, kk);
            out.push_back(this->as_int(fv));
        }
    }
    return out;
}

// The node's title address, bvh::node_address: the descriptor's base (dword 0 and the low byte of dword 1,
// in 256-byte units) plus `(pointer >> 3) << 6`.
std::pair<V, V> Lower::bvh_address(V word0, V word1, V pointer, std::optional<V> pointer_high) {
    const std::size_t c = this->cur;
    const Ty i64t = this->s.m.int_(64);
    const V zero = this->s.m.i32(0);
    const V byte = this->s.m.i32(0xff);
    const V base_high = this->s.m.bin(c, Bin::And, word1, byte);
    V base = this->join64(word0, base_high);
    const V k8 = this->s.m.int_const(i64t, 8);
    base = this->s.m.bin(c, Bin::Shl, base, k8);
    V p = this->join64(pointer, pointer_high.value_or(zero));
    const V k3 = this->s.m.int_const(i64t, 3);
    const V k6 = this->s.m.int_const(i64t, 6);
    p = this->s.m.bin(c, Bin::LShr, p, k3);
    p = this->s.m.bin(c, Bin::Shl, p, k6);
    const V a = this->s.m.bin(c, Bin::Add, base, p);
    this->s.flags |= flags::INT64;
    return this->split64(a);
}

// The box test (bvh::boxes): four result dwords. Starts in the current block and leaves `this->cur` at the
// block it ends in.
std::array<V, 4> Lower::bvh_boxes(NodeAt at, V kind, const Ray& ray, bool sort) {
    const std::size_t head = this->cur;
    const std::vector<V> children = this->bvh_node_words(at, std::nullopt, 0, 4);
    const V k5 = this->s.m.i32(5);
    const V wide = this->s.m.cmp(head, pred::EQ, kind, k5);
    std::size_t wide_block = this->s.m.new_block();
    const std::size_t half_block = this->s.m.new_block();
    const std::size_t join = this->s.m.new_block();
    this->s.m.terminate(head, Term::cond_br(wide, wide_block, half_block));
    // 32-bit floats: 24 dwords from byte 16.
    this->cur = wide_block;
    const V k16 = this->s.m.i32(16);
    std::vector<V> words = this->bvh_node_words(at, std::nullopt, 16, 12);
    const NodeAt second = this->bvh_second_half(at);
    {
        const std::vector<V> more = this->bvh_node_words(second, std::nullopt, 0, 12);
        words.insert(words.end(), more.begin(), more.end());
    }
    std::vector<V> floats;
    floats.reserve(words.size());
    for (const V w : words) {
        floats.push_back(this->as_float(w));
    }
    // (The look for the second half goes on in another block.)
    wide_block = this->cur;
    this->s.m.terminate(wide_block, Term::br(join));
    // 16-bit floats: 24 halves in 12 dwords from byte 16, the low half of a dword first.
    this->cur = half_block;
    const std::vector<V> hwords = this->bvh_node_words(at, std::nullopt, 16, 12);
    const Ty f32t = this->s.t.f32;
    std::vector<V> halves;
    halves.reserve(24);
    for (std::size_t n = 0; n < 24; ++n) {
        V h = hwords.at(n / 2);
        if (n % 2 == 1) {
            h = this->s.m.bin(half_block, Bin::LShr, h, k16);
        }
        // The conversion reads the low 16 bits.
        halves.push_back(
            this->s.op(half_block, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{h}));
    }
    this->s.m.terminate(half_block, Term::br(join));
    this->cur = join;
    // A page that is not there reads zeros, which are a child and a box: no child of it is hit.
    // (The last two boxes of a 128-byte node are in its second half.)
    std::optional<std::array<V, 4>> there;
    if (at.kind == NodeAt::Kind::Memory && second.kind == NodeAt::Kind::Memory) {
        const V first = at.mapped;
        const V last = second.mapped;
        const Ty i1t = this->s.m.type_of(first);
        const V mapped = this->s.m.phi(join, i1t);
        this->s.m.set_phi(mapped, std::vector<std::pair<V, std::size_t>>{{last, wide_block}, {first, half_block}});
        const V both = this->s.m.bin(join, Bin::And, first, mapped);
        there = std::array<V, 4>{first, first, both, both};
    }
    std::vector<V> coords;
    coords.reserve(24);
    for (std::size_t n = 0; n < 24; ++n) {
        const V phi = this->s.m.phi(join, f32t);
        this->s.m.set_phi(phi, std::vector<std::pair<V, std::size_t>>{{floats.at(n), wide_block}, {halves.at(n), half_block}});
        coords.push_back(phi);
    }
    // The inverse direction held finite (bvh::boxes).
    const V most = this->s.m.f32(f32_bits(std::numeric_limits<float>::max()));
    const V least = this->s.m.f32(f32_bits(-std::numeric_limits<float>::max()));
    std::array<V, 3> inverse{};
    for (std::size_t a = 0; a < 3; ++a) {
        const V xv = this->binary(35, ray.inverse[a], least);
        inverse[a] = this->binary(36, xv, most);
    }
    const std::size_t c = join;
    const V none = this->s.m.i32(bvh::NONE);
    const V infinity = this->s.m.f32(f32_bits(std::numeric_limits<float>::infinity()));
    const V fzero = this->s.m.f32(0);
    std::array<V, 4> ids{none, none, none, none};
    std::array<V, 4> distances{infinity, infinity, infinity, infinity};
    for (std::size_t n = 0; n < 4; ++n) {
        const V* bx = &coords[6 * n];
        std::array<V, 3> near_{fzero, fzero, fzero};
        std::array<V, 3> far_{fzero, fzero, fzero};
        for (std::size_t a = 0; a < 3; ++a) {
            V low = this->s.m.bin(c, Bin::Sub, bx[a], ray.origin[a]);
            low = this->s.m.bin(c, Bin::Mul, low, inverse[a]);
            V high = this->s.m.bin(c, Bin::Sub, bx[3 + a], ray.origin[a]);
            high = this->s.m.bin(c, Bin::Mul, high, inverse[a]);
            near_[a] = this->binary(36, low, high);
            far_[a] = this->binary(35, low, high);
        }
        V enter = this->binary(35, near_[0], near_[1]);
        enter = this->binary(35, enter, near_[2]);
        V leave = this->binary(36, far_[0], far_[1]);
        leave = this->binary(36, leave, far_[2]);
        const V from = this->binary(35, enter, fzero);
        // Ordered compares: a NaN anywhere is a miss.
        const V through = this->s.m.cmp(c, pred::OGE, leave, from);
        const V before = this->s.m.cmp(c, pred::OLT, enter, ray.extent);
        const V off = this->is_nan_f32(bx[0]);
        V on = this->s.m.bool_(true);
        on = this->s.m.bin(c, Bin::Xor, off, on);
        V hit = this->s.m.bin(c, Bin::And, through, before);
        hit = this->s.m.bin(c, Bin::And, hit, on);
        const V child = this->s.m.cmp(c, pred::NE, children.at(n), none);
        hit = this->s.m.bin(c, Bin::And, hit, child);
        if (there) {
            hit = this->s.m.bin(c, Bin::And, hit, (*there)[n]);
        }
        ids[n] = this->s.m.select(c, hit, children.at(n), none);
        distances[n] = this->s.m.select(c, hit, enter, infinity);
    }
    if (sort) {
        static constexpr std::array<std::pair<std::size_t, std::size_t>, 5> NET{
            {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}}};
        for (const auto& [a, bb] : NET) {
            const V swap = this->s.m.cmp(c, pred::OLT, distances[bb], distances[a]);
            const V da = distances[a];
            const V db = distances[bb];
            distances[a] = this->s.m.select(c, swap, db, da);
            distances[bb] = this->s.m.select(c, swap, da, db);
            const V ia = ids[a];
            const V ib = ids[bb];
            ids[a] = this->s.m.select(c, swap, ib, ia);
            ids[bb] = this->s.m.select(c, swap, ia, ib);
        }
    }
    return ids;
}

// The triangle test (bvh::triangle of bvh::triangle_vertices): four result dwords, made in the current
// block.
std::array<V, 4> Lower::bvh_triangle(NodeAt at, V kind, const Ray& ray) {
    const std::size_t c = this->cur;
    auto k = [](Lower& l, std::uint32_t v) { return l.s.m.i32(v); };
    // The triangle's byte of the dword at byte 60: which of its vertices barycentric I and J weigh.
    V id = this->bvh_node_words(at, std::nullopt, 60, 1).at(0);
    const V k2 = k(*this, 2);
    const V k3 = k(*this, 3);
    const V k12 = k(*this, 12);
    const V k15 = k(*this, 15);
    const V shift = this->s.m.bin(c, Bin::Shl, kind, k3);
    id = this->s.m.bin(c, Bin::LShr, id, shift);
    const V i = this->s.m.bin(c, Bin::And, id, k3);
    V j = this->s.m.bin(c, Bin::LShr, id, k2);
    j = this->s.m.bin(c, Bin::And, j, k3);
    V first = this->s.m.bin(c, Bin::Sub, k3, i);
    first = this->s.m.bin(c, Bin::Sub, first, j);
    first = this->s.m.bin(c, Bin::And, first, k3);
    // The node's vertices the triangle is made of, four bits each.
    const V k0 = k(*this, 0);
    const V k1 = k(*this, 1);
    std::array<V, 4> maps{};
    for (std::uint32_t n = 0; n < 4; ++n) {
        maps[n] = k(*this, bvh::vertex_map(n));
    }
    const V is0 = this->s.m.cmp(c, pred::EQ, kind, k0);
    const V is1 = this->s.m.cmp(c, pred::EQ, kind, k1);
    const V is2 = this->s.m.cmp(c, pred::EQ, kind, k2);
    V map = this->s.m.select(c, is2, maps[2], maps[3]);
    map = this->s.m.select(c, is1, maps[1], map);
    map = this->s.m.select(c, is0, maps[0], map);
    std::vector<std::array<V, 3>> p;
    p.reserve(3);
    for (const V n : {first, i, j}) {
        const V nibble = this->s.m.bin(c, Bin::Shl, n, k2);
        V vertex = this->s.m.bin(c, Bin::LShr, map, nibble);
        vertex = this->s.m.bin(c, Bin::And, vertex, k15);
        const V byte = this->s.m.bin(c, Bin::Mul, vertex, k12);
        const std::vector<V> words = this->bvh_node_words(at, byte, 0, 3);
        // Relative to the ray's origin.
        std::array<V, 3> r{};
        for (std::size_t a = 0; a < 3; ++a) {
            const V fv = this->as_float(words.at(a));
            r[a] = this->s.m.bin(c, Bin::Sub, fv, ray.origin[a]);
        }
        p.push_back(r);
    }
    // The ray's largest axis z, and x and y after it in an order that keeps the winding.
    const V fzero = this->s.m.f32(0);
    std::array<V, 3> size{};
    for (std::size_t a = 0; a < 3; ++a) {
        size[a] = this->unary(6, ray.direction[a]);
    }
    const V xy = this->s.m.cmp(c, pred::OGE, size[0], size[1]);
    const V xz = this->s.m.cmp(c, pred::OGE, size[0], size[2]);
    const V yz = this->s.m.cmp(c, pred::OGE, size[1], size[2]);
    const V z0 = this->s.m.bin(c, Bin::And, xy, xz);
    const V on = this->s.m.bool_(true);
    const V not_xy = this->s.m.bin(c, Bin::Xor, xy, on);
    const V z1 = this->s.m.bin(c, Bin::And, not_xy, yz);
    // Component (kz + n) % 3 of `q`.
    auto pick = [c, z0, z1](Lower& l, const std::array<V, 3>& q, std::size_t n) {
        const V rest = l.s.m.select(c, z1, q[(1 + n) % 3], q[(2 + n) % 3]);
        return l.s.m.select(c, z0, q[n % 3], rest);
    };
    const V dz = pick(*this, ray.direction, 0);
    const V backwards = this->s.m.cmp(c, pred::OLT, dz, fzero);
    // (x, y, z) of `q`: the two after the largest axis, swapped for a ray that runs backwards
    // along it.
    auto axes = [&pick, c, backwards](Lower& l, const std::array<V, 3>& q) {
        const V xx = pick(l, q, 1);
        const V yy = pick(l, q, 2);
        const V a0 = l.s.m.select(c, backwards, yy, xx);
        const V a1 = l.s.m.select(c, backwards, xx, yy);
        const V a2 = pick(l, q, 0);
        return std::array<V, 3>{a0, a1, a2};
    };
    const std::array<V, 3> d = axes(*this, ray.direction);
    const V sz = pick(*this, ray.inverse, 0);
    const V sx = this->s.m.bin(c, Bin::Mul, d[0], sz);
    const V sy = this->s.m.bin(c, Bin::Mul, d[1], sz);
    std::vector<std::array<V, 3>> sheared;
    sheared.reserve(3);
    for (const auto& r0 : p) {
        const std::array<V, 3> r = axes(*this, r0);
        V xv = this->s.m.bin(c, Bin::Mul, sx, r[2]);
        xv = this->s.m.bin(c, Bin::Sub, r[0], xv);
        V yv = this->s.m.bin(c, Bin::Mul, sy, r[2]);
        yv = this->s.m.bin(c, Bin::Sub, r[1], yv);
        const V zv = this->s.m.bin(c, Bin::Mul, sz, r[2]);
        sheared.push_back({xv, yv, zv});
    }
    if (sheared.size() != 3) std::abort();  // unreachable
    const V ax = sheared[0][0], ay = sheared[0][1], az = sheared[0][2];
    const V bx = sheared[1][0], by = sheared[1][1], bz = sheared[1][2];
    const V cx = sheared[2][0], cy = sheared[2][1], cz = sheared[2][2];
    // p * q - r * s.
    auto cross = [c](Lower& l, V pp, V q, V r, V s) {
        const V a = l.s.m.bin(c, Bin::Mul, pp, q);
        const V bb = l.s.m.bin(c, Bin::Mul, r, s);
        return l.s.m.bin(c, Bin::Sub, a, bb);
    };
    const V u = cross(*this, cx, by, cy, bx);
    const V v = cross(*this, ax, cy, ay, cx);
    const V w = cross(*this, bx, ay, by, ax);
    auto any = [c, u, v, w, fzero](Lower& l, std::uint8_t pr) {
        V a = l.s.m.cmp(c, pr, u, fzero);
        const V bb = l.s.m.cmp(c, pr, v, fzero);
        const V dd = l.s.m.cmp(c, pr, w, fzero);
        a = l.s.m.bin(c, Bin::Or, a, bb);
        return l.s.m.bin(c, Bin::Or, a, dd);
    };
    const V behind = any(*this, pred::OLT);
    const V before = any(*this, pred::OGT);
    const V outside = this->s.m.bin(c, Bin::And, behind, before);
    const V inside = this->s.m.bin(c, Bin::Xor, outside, on);
    V det = this->s.m.bin(c, Bin::Add, v, w);
    det = this->s.m.bin(c, Bin::Add, u, det);
    V t = this->s.m.bin(c, Bin::Mul, u, az);
    V t2 = this->s.m.bin(c, Bin::Mul, v, bz);
    t = this->s.m.bin(c, Bin::Add, t, t2);
    t2 = this->s.m.bin(c, Bin::Mul, w, cz);
    t = this->s.m.bin(c, Bin::Add, t, t2);
    // The distance is not negative: the numerator has the denominator's sign (or is 0). Ordered
    // compares: a denominator of 0 or a NaN is a miss.
    V front = this->s.m.cmp(c, pred::OGT, det, fzero);
    const V front_ahead = this->s.m.cmp(c, pred::OGE, t, fzero);
    front = this->s.m.bin(c, Bin::And, front, front_ahead);
    V back = this->s.m.cmp(c, pred::OLT, det, fzero);
    const V back_ahead = this->s.m.cmp(c, pred::OLE, t, fzero);
    back = this->s.m.bin(c, Bin::And, back, back_ahead);
    const V ahead = this->s.m.bin(c, Bin::Or, front, back);
    const V hit = this->s.m.bin(c, Bin::And, inside, ahead);
    std::array<V, 4> miss{};
    for (std::size_t n = 0; n < 4; ++n) {
        miss[n] = this->s.m.f32(f32_bits(bvh::MISS[n]));
    }
    const std::array<V, 4> out{t, det, v, w};
    std::array<V, 4> result{};
    for (std::size_t n = 0; n < 4; ++n) {
        const V fv = this->s.m.select(c, hit, out[n], miss[n]);
        result[n] = this->as_int(fv);
    }
    return result;
}

}  // namespace x2w::vs::lower
