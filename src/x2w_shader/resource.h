// Resource tracking: where each descriptor a program uses comes from, in terms of the root signature.
//
// Root parameters arrive in user SGPRs (RootParam): a descriptor table as the 32-bit heap address of
// its range (the code sets the high half to 0), a root descriptor as a 64-bit buffer address, root
// constants as themselves. A root signature with dynamic resources (SM 6.6) adds the 32-bit base
// addresses of both heaps; a descriptor is then loaded from one of them at an index the code
// computes (from a root constant, say). The code loads table descriptors from heap memory with
// `s_load_dword*`, and builds a buffer descriptor around a root descriptor's address itself. Each
// descriptor operand and each scalar memory address is traced through the SSA values back to those
// parameters, by data flow: copies, phis that agree, and additions.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "x2w_shader/error.h"
#include "x2w_shader/front_hash.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/structure.h"

namespace x2w::vs::resource {

// An instruction: block and index in ir::Block::insts.
using Site = std::pair<std::size_t, std::size_t>;
using SiteHash = fhash::SiteHash;

// A byte address: a root parameter's, plus a constant and at most one value the program computes
// (an index times a stride, say).
struct Address {
    // Index into the layout, or CODE (or MEMORY).
    std::size_t param = 0;
    // Added modulo 2^32 (with carry into the high half of a root descriptor's address).
    std::uint32_t offset = 0;
    std::optional<ir::Value> dynamic;
    bool operator==(const Address&) const = default;
};

// Address::param of the program's own code: a constant table the compiler appends after the code
// and reads through `s_getpc_b64`; the offset is then the byte offset in the code.
inline constexpr std::size_t CODE = SIZE_MAX;

// Address::param of title memory at an address no root parameter gives: one the code computes from
// root constants, reads out of a descriptor or loads from other memory (compute programs only: the
// ray-tracing structure builder and the programs that walk its structures). Offset and computed term
// are not used: lowering takes the 64-bit address from the instruction's operands.
inline constexpr std::size_t MEMORY = SIZE_MAX - 1;

// Leave fields unused by the active source kind empty.
struct Source {
    enum class Kind : std::uint8_t {
        // Dwords `first..` of what the scalar load at `load` reads (see Resources::scalar for its
        // address).
        Loaded,
        // A buffer descriptor built around an address: `bits` OR-ed into dword 1 (stride, swizzle),
        // dwords 2 and 3 as given (`words`).
        Built,
        // Fixed in the code (an immediate sampler, say): `constant`.
        Constant,
        // Chosen by control flow: one source per predecessor of `block` (`args`), in
        // ir::Block::preds order.
        Phi,
        // Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each dword reads): the source
        // when it is set, then when it is clear (`args`).
        Select,
        // Dwords `first..` of what the vector load at `load` (`global_load_*`) reads, each lane at its
        // own address (see Resources::scalar); the dwords past that load come from loads of the bytes
        // that follow. A descriptor that differs between lanes.
        Fetched,
        // Picked per lane by `v_cndmask_b32` reading the lane mask whose low dword is `mask`: the
        // source where the lane's bit is set, then where it is clear (`args`).
        Lanes,
        // Per-lane dwords chosen by control flow (VGPRs joined where lanes that took different paths
        // meet): one source per predecessor of `block` (`lane_args`), in ir::Block::preds order,
        // nullopt on an edge no lane that accesses the descriptor takes. A descriptor that differs
        // between lanes.
        LanePhi,
    };
    Kind kind = Kind::Constant;
    Site load{};                         // Loaded, Fetched
    std::uint8_t first = 0;              // Loaded, Fetched
    Address address;                     // Built
    std::uint32_t bits = 0;              // Built
    std::array<std::uint32_t, 2> words{};  // Built
    std::vector<std::uint32_t> constant;  // Constant
    std::size_t block = 0;               // Phi, LanePhi
    ir::Value scc;                       // Select
    ir::Value mask;                      // Lanes
    std::vector<Source> args;            // Phi, Select, Lanes
    std::vector<std::optional<Source>> lane_args;  // LanePhi

    static Source Loaded(Site load, std::uint8_t first);
    static Source Built(Address address, std::uint32_t bits, std::array<std::uint32_t, 2> words);
    static Source Constant(std::vector<std::uint32_t> words);
    static Source Phi(std::size_t block, std::vector<Source> args);
    static Source Select(ir::Value scc, std::vector<Source> args);
    static Source Fetched(Site load, std::uint8_t first);
    static Source Lanes(ir::Value mask, std::vector<Source> args);
    static Source LanePhi(std::size_t block, std::vector<std::optional<Source>> args);

    // The sources this one picks from (itself unless a phi).
    std::vector<const Source*> leaves() const;
    bool operator==(const Source&) const;
};

enum class Role : std::uint8_t {
    // Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
};

struct Descriptor {
    Site site{};
    Role role = Role::Resource;
    Source source;
    bool operator==(const Descriptor&) const = default;
};

// A view-heap descriptor whose byte address can be evaluated when binding the draw, using a table
// base and a root constant.
struct RootIndexedView {
    std::uint8_t table_slot = 0;
    std::uint8_t constant_slot = 0;
    std::uint32_t constant_dword = 0;
    std::uint32_t byte_offset = 0;
    std::uint8_t shift = 0;
    bool operator==(const RootIndexedView&) const = default;
};

// A ray-tracing node test (bvh.h) addresses a range of memory containing nodes.
struct BvhNodes {
    enum class Kind : std::uint8_t {
        // A buffer of the host: the descriptor is loaded from a table (it is among
        // Resources::descriptors), and node pointers count from the buffer's start. Its flags are not
        // known to the shader: boxes are sorted and triangles return barycentrics, as every
        // descriptor the title builds asks.
        Resource,
        // Title memory by address: the code makes the descriptor from numbers (the title: the whole
        // address space, or one structure's address in dword 0). Dwords 1 and 3, which hold its
        // flags.
        Memory,
    };
    Kind kind = Kind::Resource;
    std::uint32_t word1 = 0;  // Memory
    std::uint32_t word3 = 0;  // Memory
    static constexpr BvhNodes Resource() { return {Kind::Resource, 0, 0}; }
    static constexpr BvhNodes Memory(std::uint32_t w1, std::uint32_t w3) { return {Kind::Memory, w1, w3}; }
    bool operator==(const BvhNodes&) const = default;
};

// Waterfall loops over descriptors that differ between lanes: each pass reads the first active
// lane's descriptor with `v_readfirstlane_b32`, compares it with every lane's own (`v_cmp_eq`), and
// runs the access with the lanes that match. Their accesses take each lane's own descriptor
// (Source::Fetched, Source::Lanes), so the first lane's dwords are each lane's own too: the compares
// hold in every lane and one pass serves all lanes.
struct Waterfall {
    // The compares, true in every lane.
    std::vector<Site> compares;
    // The `v_readfirstlane_b32` results they compare, each the lane's own source dword.
    std::vector<ir::Value> reads;
    // Descriptor loads at a table offset among `reads` (the console compiler's form): each lane loads
    // its own entry.
    std::vector<Site> loads;
    bool operator==(const Waterfall&) const = default;
};

struct Resources {
    // Every scalar memory access by address (`s_load_*`, `s_store_*`), and every global memory
    // access (`global_*`, the dynamic term per lane).
    std::vector<std::pair<Site, Address>> scalar;
    std::vector<Descriptor> descriptors;
    // Direct descriptor sources with a draw-evaluable index.
    std::vector<RootIndexedView> root_indexed_views;
    // User SGPRs holding root parameters: value on entry, layout index, dword within the parameter.
    std::vector<std::tuple<ir::Value, std::size_t, std::uint32_t>> params;
    // Scalar loads of one descriptor the code keeps per lane in VGPRs: they read one address, so
    // their dwords stand in for each other.
    std::vector<Site> lane_copied;
    Waterfall waterfall;
    // Every `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`: where its nodes are.
    std::vector<std::pair<Site, BvhNodes>> bvh;
    bool operator==(const Resources&) const = default;
};

// Traces every descriptor and scalar address of `p`, which runs as `stage`, to the root parameters
// in `layout`. Throws ShaderError on the first one it cannot trace. `lanes`, `p` as one lane runs it,
// tells which paths the lanes that access a per-lane descriptor took and which writes they ran;
// without it (nullptr), per-lane descriptors chosen by control flow are refused and per-lane writes
// are taken to run in every lane.
Resources track(const ir::Program& p, std::span<const native::RootParam> layout, native::HwStage stage,
                const structure::Structured* lanes);

// The 32-bit value of an inline constant operand.
std::optional<std::uint32_t> inline_constant(std::uint8_t code);

namespace detail {
// A dword after copy propagation.
struct Term {
    enum class Kind : std::uint8_t { Const, Value };
    Kind kind = Kind::Const;
    std::uint32_t n = 0;
    static constexpr Term Const(std::uint32_t c) { return {Kind::Const, c}; }
    static constexpr Term Of(ir::Value v) { return {Kind::Value, v.n}; }
    constexpr bool is_const() const { return kind == Kind::Const; }
    constexpr bool is_value() const { return kind == Kind::Value; }
    constexpr ir::Value value() const { return {n}; }
    constexpr bool operator==(const Term&) const = default;
};
// Every value's origin after copy propagation (Tracer::new(..).origin, for the tests). Throws
// ShaderError.
std::vector<Term> origins(const ir::Program& p, std::span<const native::RootParam> layout, native::HwStage stage);
}  // namespace detail

}  // namespace x2w::vs::resource
