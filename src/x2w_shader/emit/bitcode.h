#pragma once

// A DXIL module and its LLVM 3.7 bitcode, written with unabbreviated records only (the format DXIL fixes;
// codes from LLVM's LLVMBitCodes.h).
//
// Values are symbolic while the module is built (V) and numbered when it is written: functions,
// then constants (all at module level), then the instructions of `main` in block order.
//
// Builder calls assign type, constant, function and instruction numbers. Evaluate them in separate
// statements in the intended order to keep output bytes deterministic. Internal invariant failures throw
// std::logic_error. The trailing underscores on predicate names avoid the macros in windows.h.

#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace x2w::vs::bitcode {

namespace census {
struct Census;
}
namespace opt {
struct Usage;
enum class Effect : std::uint8_t;
}

// Signals a translator invariant failure.
[[noreturn]] void bug(const char* what);

struct Ty {
    std::uint32_t id = 0;
    auto operator<=>(const Ty&) const = default;
};

// What the lowering made an instruction for, where it says so (counted by census, nothing else
// reads it).
enum class Tag : std::uint8_t {
    // The select that keeps a register's old value in lanes a guard masks.
    Guard,
    // An index clamp the translation adds (the console has none).
    Clamp,
    // The iteration counter and limit test the translation adds to a loop.
    Counter,
};

struct TypeDef {
    enum Kind : std::uint8_t { Void, Int, Float, Double, Ptr, Struct, Array, Func };
    Kind kind = Void;
    // Int: bits. Ptr: address space. Array: element count.
    std::uint32_t n = 0;
    // Ptr: pointee. Array: element. Func: return type.
    Ty of;
    // Struct: named (Some) or literal (None).
    std::optional<std::string> name;
    // Struct: fields. Func: parameters.
    std::vector<Ty> list;
    bool operator==(const TypeDef&) const = default;
};

// Order values by kind (Global < Func < Const < Inst), then by index.
struct V {
    enum Kind : std::uint8_t { Global, Func, Const, Inst };
    Kind kind = Const;
    std::uint32_t i = 0;
    static constexpr V global(std::uint32_t i) { return V{Global, i}; }
    static constexpr V func(std::uint32_t i) { return V{Func, i}; }
    static constexpr V constant(std::uint32_t i) { return V{Const, i}; }
    static constexpr V inst(std::uint32_t i) { return V{Inst, i}; }
    bool is_inst() const { return kind == Inst; }
    auto operator<=>(const V&) const = default;
};

struct VHash {
    std::size_t operator()(const V& v) const noexcept {
        return std::hash<std::uint64_t>{}((std::uint64_t(v.kind) << 32) | v.i);
    }
};

// A global variable declaration (thread-group shared memory, constant tables).
struct Global {
    std::string name;
    Ty ty;
    // Its pointer type.
    Ty ptr;
    std::uint32_t space = 0;
    // An internal constant with this initialiser (a constant table).
    std::optional<std::uint32_t> init;
};

struct Const {
    enum Kind : std::uint8_t { Int, Float, Undef, Aggregate };
    Kind kind = Undef;
    std::int64_t value = 0;   // Int
    std::uint32_t bits = 0;   // Float
    std::vector<std::uint32_t> ids;  // Aggregate: constant ids
    bool operator==(const Const&) const = default;
};

// LLVM binary opcodes (the same codes serve the float forms).
enum class Bin : std::uint8_t {
    Add = 0,
    Sub = 1,
    Mul = 2,
    UDiv = 3,
    SDiv = 4,
    URem = 5,
    Shl = 7,
    LShr = 8,
    AShr = 9,
    And = 10,
    Or = 11,
    Xor = 12,
};

// atomicrmw operations.
enum class Rmw : std::uint8_t {
    Xchg = 0,
    Add = 1,
    Sub = 2,
    And = 3,
    Or = 5,
    Xor = 6,
    Max = 7,
    Min = 8,
    UMax = 9,
    UMin = 10,
};

enum class Cast : std::uint8_t {
    Trunc = 0,
    ZExt = 1,
    SExt = 2,
    FpToUi = 3,
    FpToSi = 4,
    UiToFp = 5,
    SiToFp = 6,
    Bitcast = 11,
};

// Compare predicates: floats 0-15, integers 32-41.
namespace pred {
inline constexpr std::uint8_t FALSE_ = 0;
inline constexpr std::uint8_t OEQ = 1;
inline constexpr std::uint8_t OGT = 2;
inline constexpr std::uint8_t OGE = 3;
inline constexpr std::uint8_t OLT = 4;
inline constexpr std::uint8_t OLE = 5;
// Not emitted: the AMD D3D12 driver treats it as UNE, so v_cmp_lg lowers as OLT | OGT instead.
inline constexpr std::uint8_t ONE = 6;
inline constexpr std::uint8_t ORD = 7;
inline constexpr std::uint8_t UNO = 8;
inline constexpr std::uint8_t UEQ = 9;
inline constexpr std::uint8_t UGT = 10;
inline constexpr std::uint8_t UGE = 11;
inline constexpr std::uint8_t ULT = 12;
inline constexpr std::uint8_t ULE = 13;
inline constexpr std::uint8_t UNE = 14;
inline constexpr std::uint8_t TRUE_ = 15;
inline constexpr std::uint8_t EQ = 32;
inline constexpr std::uint8_t NE = 33;
inline constexpr std::uint8_t IUGT = 34;
inline constexpr std::uint8_t IUGE = 35;
inline constexpr std::uint8_t IULT = 36;
inline constexpr std::uint8_t IULE = 37;
inline constexpr std::uint8_t SGT = 38;
inline constexpr std::uint8_t SGE = 39;
inline constexpr std::uint8_t SLT = 40;
inline constexpr std::uint8_t SLE = 41;
}  // namespace pred

// An instruction's operation. Which fields hold what, by kind:
//   Bin       code = Bin, a, b
//   Cast      code = Cast, a
//   Cmp       code = predicate, a, b
//   Select    a = condition, b = true value, c = false value
//   Extract   a = aggregate, index = field
//   Call      index = function number, list = arguments
//   Phi       incoming = (value, block); filled in later
//   Gep       ty = pointee type, a = pointer, list = indices
//   Load      a = pointer
//   Store     a = pointer, b = value
//   AtomicRmw code = Rmw, a = pointer, b = value
//   CmpXchg   a = pointer, b = expected, c = new; returns {i32, i1}
struct Op {
    enum Kind : std::uint8_t { Bin, Cast, Cmp, Select, Extract, Call, Phi, Gep, Load, Store, AtomicRmw, CmpXchg };
    Kind kind = Bin;
    std::uint8_t code = 0;
    V a, b, c;
    std::uint32_t index = 0;
    Ty ty;
    std::vector<V> list;
    std::vector<std::pair<V, std::size_t>> incoming;
};

struct Inst {
    Ty ty;
    Op op;
};

struct Term {
    enum Kind : std::uint8_t { Br, CondBr, Ret };
    Kind kind = Ret;
    V cond;
    std::size_t t = 0;
    std::size_t f = 0;
    static Term br(std::size_t to) { return Term{Br, V{}, to, 0}; }
    static Term cond_br(V c, std::size_t t, std::size_t f) { return Term{CondBr, c, t, f}; }
    static Term ret() { return Term{Ret, V{}, 0, 0}; }
};

struct Block {
    std::vector<std::uint32_t> phis;
    std::vector<std::uint32_t> body;
    std::optional<Term> term;
};

// Function attributes of a dx.op declaration.
enum class Attr : std::uint8_t {
    ReadNone = 1,
    NoUnwind = 2,
    ReadOnly = 3,
    // Barriers.
    NoDuplicate = 4,
};

struct Func {
    std::string name;
    Ty ty;
    std::optional<Attr> attr;
    bool body = false;
};

// A metadata node operand.
struct Md {
    std::uint32_t id = 0;
    auto operator<=>(const Md&) const = default;
};

struct MdDef {
    enum Kind : std::uint8_t { Str, Value, Node };
    Kind kind = Str;
    std::string str;
    Ty ty;
    V v;
    std::vector<std::optional<Md>> ops;
};

struct TypeDefHash {
    std::size_t operator()(const TypeDef& t) const noexcept;
};
struct ConstKeyHash {
    std::size_t operator()(const std::pair<Ty, Const>& k) const noexcept;
};

// A block's successors (at most two).
struct Succs {
    std::size_t to[2] = {0, 0};
    std::size_t len = 0;
    const std::size_t* begin() const { return to; }
    const std::size_t* end() const { return to + len; }
    std::size_t size() const { return len; }
    bool empty() const { return len == 0; }
    bool contains(std::size_t b) const {
        for (std::size_t k = 0; k < len; ++k)
            if (to[k] == b) return true;
        return false;
    }
};

Succs successors(const std::optional<Term>& term);

// Calls f with every value op reads, in operand order.
template <class F>
void for_each_operand(const Op& op, F&& f) {
    switch (op.kind) {
    case Op::Bin:
    case Op::Cmp:
    case Op::Store:
    case Op::AtomicRmw:
        f(op.a);
        f(op.b);
        break;
    case Op::Cast:
    case Op::Extract:
    case Op::Load:
        f(op.a);
        break;
    case Op::Select:
    case Op::CmpXchg:
        f(op.a);
        f(op.b);
        f(op.c);
        break;
    case Op::Call:
        for (V v : op.list) f(v);
        break;
    case Op::Phi:
        for (const auto& x : op.incoming) f(x.first);
        break;
    case Op::Gep:
        f(op.a);
        for (V v : op.list) f(v);
        break;
    }
}

// The optimizer and instruction census access the module members directly.
struct Module {
    std::vector<TypeDef> types;
    std::unordered_map<TypeDef, std::uint32_t, TypeDefHash> type_ids;
    std::vector<std::pair<Ty, Const>> consts;
    std::unordered_map<std::pair<Ty, Const>, std::uint32_t, ConstKeyHash> const_ids;
    std::vector<Global> globals;
    std::vector<Func> funcs;
    std::unordered_map<std::string, std::uint32_t> func_ids;
    std::vector<Inst> insts;
    std::vector<Block> blocks;
    std::vector<MdDef> mds;
    std::vector<std::pair<std::string, std::vector<Md>>> named;
    // See Tag: by instruction.
    std::unordered_map<std::uint32_t, Tag> tags;

    Ty ty(const TypeDef& def);
    Ty void_();
    Ty int_(std::uint32_t bits);
    Ty float_();
    Ty double_();
    Ty ptr(Ty to);
    Ty ptr_in(Ty to, std::uint32_t space);
    Ty array(std::uint32_t n, Ty of);
    Ty named_struct(std::string_view name, std::vector<Ty> fields);
    Ty func_ty(Ty ret, std::vector<Ty> params);
    bool is_void(Ty t) const;

    V constant(Ty ty, const Const& c);
    V int_const(Ty ty, std::int64_t v);
    V i32(std::uint32_t v);
    V i8(std::uint8_t v);
    V bool_(bool v);
    V f32(std::uint32_t bits);
    V undef(Ty ty);
    // A constant struct or array of constants.
    V aggregate(Ty ty, std::span<const V> fields);
    V aggregate(Ty ty, std::initializer_list<V> fields) {
        return aggregate(ty, std::span<const V>(fields.begin(), fields.size()));
    }
    // The value of an integer constant; nullopt for anything else.
    std::optional<std::int64_t> int_value(V v) const;
    // Gives constant `old` the content of `new_` (both stay valid).
    void replace_const(V old, V new_);

    // A function declaration (or `main` with `body`), found by name.
    V function(std::string_view name, Ty ty, std::optional<Attr> attr, bool body);
    // An external global of type `ty` in address space `space` (DXIL declares thread-group shared
    // memory this way).
    V global(std::string_view name, Ty ty, std::uint32_t space);
    // An internal constant global initialised with the constant `init`.
    V constant_global(std::string_view name, V init);
    // Notes what instruction v is for (see Tag); only when modules are counted.
    void tag(V v, Tag tag);

    std::size_t new_block();
    Ty type_of(V v) const;

    V push(std::size_t block, Ty ty, Op op);
    V bin(std::size_t block, Bin op, V a, V b);
    V cast(std::size_t block, Cast op, V v, Ty to);
    V cmp(std::size_t block, std::uint8_t pred, V a, V b);
    V select(std::size_t block, V c, V t, V f);
    V extract(std::size_t block, V agg, std::uint32_t index);
    V call(std::size_t block, V f, std::vector<V> args);
    // A phi at the top of block; incoming values come with set_phi.
    V phi(std::size_t block, Ty ty);
    // The address of element index of the array ptr points to.
    V element(std::size_t block, V ptr, V index);
    V load(std::size_t block, V ptr);
    void store(std::size_t block, V ptr, V value);
    V atomic(std::size_t block, Rmw op, V ptr, V value);
    // The old value of a compare-exchange.
    V cmpxchg(std::size_t block, V ptr, V expected, V new_);

    void set_phi(V phi, std::vector<std::pair<V, std::size_t>> incoming);
    void terminate(std::size_t block, Term t);

    Md md_str(std::string_view s);
    Md md_value(V v);
    Md md_node(std::vector<std::optional<Md>> ops);
    void named_md(std::string_view name, std::vector<Md> nodes);

    // Which instructions are written: every body instruction, the branch conditions, and whatever
    // they read, transitively (a phi nothing reads is dropped).
    std::vector<bool> live() const;
    // The module as bitcode.
    std::vector<std::uint8_t> write() const;

    // opt.cpp. Runs the clean-up passes named in `passes` (opt::ALL: every one).
    void optimise(std::uint32_t passes);
    // Per function: what a call of it does.
    std::vector<opt::Effect> effects() const;
    // The blocks reached from the entry.
    std::vector<bool> reached() const;
    // An integer constant: its width and its bits.
    std::optional<std::pair<std::uint32_t, std::uint64_t>> int_k(V v) const;
    // A float constant's value.
    std::optional<float> float_k(V v) const;
    // A bound on |v| (see opt.cpp).
    std::optional<float> bounded(V v, std::uint32_t depth) const;
    // What the written instructions use, of the features a shader declares by flag.
    opt::Usage usage() const;

    // census.cpp: the instructions write() writes, by kind.
    census::Census census() const;
};

// LLVM's signed VBR operand: magnitude shifted left, sign in bit 0.
std::uint64_t signed_vbr(std::int64_t v);

// The bitstream: little-endian 32-bit words filled from bit 0.
struct Writer {
    std::vector<std::uint32_t> words;
    std::uint64_t cur = 0;
    std::uint32_t bits = 0;
    // Per open block: abbreviation width, index of its length word.
    std::vector<std::pair<std::uint32_t, std::size_t>> stack;

    std::uint32_t width() const;
    void emit(std::uint64_t v, std::uint32_t width);
    void vbr(std::uint64_t v, std::uint32_t width);
    void align();
    void bytes(const std::uint8_t* b, std::size_t n);
    void enter(std::uint64_t id, std::uint32_t width);
    void exit();
    // An unabbreviated record.
    void record(std::uint64_t code, const std::uint64_t* ops, std::size_t n);
    void record(std::uint64_t code, const std::vector<std::uint64_t>& ops) {
        record(code, ops.data(), ops.size());
    }
    void record(std::uint64_t code, std::initializer_list<std::uint64_t> ops) {
        record(code, ops.begin(), ops.size());
    }
    std::vector<std::uint8_t> finish();
};

}  // namespace x2w::vs::bitcode
