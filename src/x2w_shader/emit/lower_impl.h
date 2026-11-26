// Emitter state and methods private to lower_*.cpp.
//
// Methods throw WhyFail for a missing value, StopRetry to retry lowering, or ShaderError for a terminal
// refusal. A false result can indicate an absent optional operation. Keep one builder call per statement
// because creation order determines type, constant and instruction numbers. Sort unordered collections
// before emitting anything that depends on their order. See GMR/vs_port/lower/NOTES.md.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/opt.h"
#include "x2w_shader/emit/opt.h"
#include "x2w_shader/emit/own_bits.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/wave.h"
#include "x2w_shader/gfx10/fields.h"
#include "x2w_shader/spill.h"
#include "x2w_shader/structure.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/io.h"
#include "x2w_shader/native.h"
#include "x2w_shader/bvh.h"
#include <algorithm>

namespace x2w::vs::lower {

// Namespaces used by the lowering implementation.
namespace emit = ::x2w::vs;
using bitcode::Attr;
using bitcode::Bin;
using bitcode::Cast;
using bitcode::Rmw;
using bitcode::Tag;
using bitcode::Term;
using bitcode::Ty;
using bitcode::V;
namespace pred = bitcode::pred;
namespace opt = bitcode::opt;
using shader::Mesh;
using shader::Shader;
namespace flags = shader::flags;
// Shader ABI constants and heap settings, including SLOT_STRIDES, USER_VIEW_BIAS, raw_views and
// view_heap_limit.
using namespace ::x2w::vs::shader;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;
using gfx10::Dpp;
using gfx10::Fields;
using gfx10::Operand;
using io::Barycentric;
using io::Export;
using io::Input;
using io::Interp;
using io::Io;
using io::Target;
using native::HwStage;
using native::NativeProgram;
using native::ResourceType;
using native::RootType;
using native::StageIo;
using native::Usage;
using native::UsedResource;
using resource::Address;
using resource::Resources;
using resource::Role;
using resource::Site;
using resource::Source;
using structure::Atom;
using structure::Bool;
using structure::LaneExit;
using structure::Structured;

// Hashing composite keys.
template <class T>
std::size_t hash_of(const T& v);

inline std::size_t hash_mix(std::size_t seed, std::size_t h) {
    return seed ^ (h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

template <class A, class B>
std::size_t hash_of_pair(const std::pair<A, B>& p) {
    return hash_mix(hash_of(p.first), hash_of(p.second));
}

template <class T>
std::size_t hash_of(const T& v) {
    if constexpr (requires { std::tuple_size<T>::value; } && !std::is_array_v<T>) {
        std::size_t seed = 0;
        std::apply([&](const auto&... e) { ((seed = hash_mix(seed, hash_of(e))), ...); }, v);
        return seed;
    } else if constexpr (std::is_same_v<T, ir::Value>) {
        return std::hash<std::uint32_t>{}(v.n);
    } else if constexpr (std::is_same_v<T, bitcode::V>) {
        return bitcode::VHash{}(v);
    } else if constexpr (std::is_same_v<T, structure::Bool>) {
        return structure::BoolHash{}(v);
    } else if constexpr (std::is_same_v<T, ir::Reg>) {
        return std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(v.kind) << 8 | v.n);
    } else if constexpr (std::is_enum_v<T>) {
        return std::hash<std::underlying_type_t<T>>{}(static_cast<std::underlying_type_t<T>>(v));
    } else {
        return std::hash<T>{}(v);
    }
}

struct Hash {
    template <class T>
    std::size_t operator()(const T& v) const {
        return hash_of(v);
    }
};

template <class K, class T>
using HashMap = std::unordered_map<K, T, Hash>;
template <class K>
using HashSet = std::unordered_set<K, Hash>;
template <class K, class T>
using BTreeMap = std::map<K, T>;

// Failures while resolving values.
struct WhyFail {
    Why why;
};
// a phi whose incoming value has no DXIL form was read: redo with it missing. (`Stop::Fail(e)` is a thrown
// ShaderError.)
struct StopRetry {
    Value value;
    Why why;
};

[[noreturn]] inline void fail(Why why) { throw WhyFail{why}; }
// Throw an unsupported instruction error with the offset and reason.
[[noreturn]] inline void unsupported(std::size_t offset, Why what) {
    throw ShaderError::unsupported(offset, what);
}
template <class T>
T ok_or(std::optional<T> o, Why why) {
    if (!o) fail(why);
    return *o;
}
template <class T>
T* ok_or(T* p, Why why) {
    if (!p) fail(why);
    return p;
}
template <class F>
auto ok(F&& f) -> std::optional<decltype(f())> {
    try {
        return f();
    } catch (const WhyFail&) {
        return std::nullopt;
    }
}
template <class F>
bool is_ok(F&& f) {
    try {
        f();
        return true;
    } catch (const WhyFail&) {
        return false;
    }
}
// Return the error from f, if any.
template <class F>
std::optional<Why> err_of(F&& f) {
    try {
        f();
    } catch (const WhyFail& e) {
        return e.why;
    }
    return std::nullopt;
}

inline bool same(Why a, Why b) { return std::string_view(a) == std::string_view(b); }

// Shift counts must be below the operand width. Assert in debug builds and mask the count in release
// builds.
inline std::uint32_t shl32(std::uint32_t x, std::uint32_t n) { return x << (n & 31); }
inline std::uint32_t shr32(std::uint32_t x, std::uint32_t n) { return x >> (n & 31); }
inline std::uint64_t shl64(std::uint64_t x, std::uint32_t n) { return x << (n & 63); }
inline std::uint64_t shr64(std::uint64_t x, std::uint32_t n) { return x >> (n & 63); }
// Sites requiring a zero for an out-of-range shift check the count explicitly.

inline std::uint32_t f32_as_u32(float f) {
    if (std::isnan(f) || f <= 0.0f) return 0;
    if (f >= 4294967296.0f) return std::numeric_limits<std::uint32_t>::max();
    return static_cast<std::uint32_t>(f);
}
inline std::int32_t f32_as_i32(float f) {
    if (std::isnan(f)) return 0;
    if (f <= -2147483648.0f) return std::numeric_limits<std::int32_t>::min();
    if (f >= 2147483648.0f) return std::numeric_limits<std::int32_t>::max();
    return static_cast<std::int32_t>(f);
}
inline std::uint32_t f32_bits(float f) { return std::bit_cast<std::uint32_t>(f); }
inline float f32_from_bits(std::uint32_t b) { return std::bit_cast<float>(b); }
// the other operand when one is NaN.
inline float rs_fmin(float a, float b) { return std::isnan(a) ? b : std::isnan(b) ? a : (a < b ? a : b); }
inline float rs_fmax(float a, float b) { return std::isnan(a) ? b : std::isnan(b) ? a : (a > b ? a : b); }
// usize::div_ceil, u32::div_ceil.
template <class T>
constexpr T div_ceil(T a, T b) {
    return a / b + (a % b != 0 ? 1 : 0);
}

// How interpolation evaluates an input channel.
struct Eval {
    enum class Kind : std::uint8_t { Mode, Weights } kind = Kind::Mode;
    // Mode: loaded with this DXIL interpolation mode (1: constant, P0).
    std::uint8_t mode = 0;
    // Weights: from the vertex values of a constant element: P0 when `p0`, plus i (P1 - P0), plus
    // j (P2 - P0), weights as floats.
    bool p0 = false;
    std::optional<V> i;
    std::optional<V> j;

    static Eval Mode(std::uint8_t m) {
        Eval e;
        e.kind = Kind::Mode;
        e.mode = m;
        return e;
    }
    static Eval Weights(bool p0, std::optional<V> i, std::optional<V> j) {
        Eval e;
        e.kind = Kind::Weights;
        e.p0 = p0;
        e.i = i;
        e.j = j;
        return e;
    }
};

struct Val {
    enum class Kind : std::uint8_t { I32, Bit, Missing } kind = Kind::Missing;
    V v{};
    Why why = "";

    static Val I32(V v) { return Val{Kind::I32, v, ""}; }
    static Val Bit(V v) { return Val{Kind::Bit, v, ""}; }
    static Val Missing(Why w) { return Val{Kind::Missing, V{}, w}; }
    bool is_i32() const { return kind == Kind::I32; }
    bool is_bit() const { return kind == Kind::Bit; }
    bool is_missing() const { return kind == Kind::Missing; }
};

// How lane masks become data.
enum class MaskData : std::uint8_t { Console, Host, None };

struct NodeAt;
struct Ray;

// Lowering helpers implemented in lower_free.cpp.
HashSet<Site> ordered_section(const Program& p);
std::uint32_t inline_f16(std::uint8_t c);
enum class Numeric : std::uint8_t { Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float };
std::optional<std::pair<std::span<const std::uint32_t>, Numeric>> buffer_format(std::uint8_t format);
std::uint32_t half_shift(std::uint32_t width);
std::optional<std::vector<bool>> all_lane_blocks(const Structured& st);
void need(std::size_t block);
HashSet<Value> lane_uniform(const Program& p, const Structured& st, const std::vector<bool>& all);
std::uint32_t meeting_mode();
std::optional<std::vector<std::pair<bool, bool>>> lds_meetings(const Program& p, const Structured& st,
                                                               const std::vector<bool>& all);
bool lane_constant_masks(const Structured& st);
bool f16_nearest_on();
bool own_bits_on();
bool local_waves_on();       // LOCAL_WAVES.get()
bool local_any_on();         // LOCAL_ANY.get()
bool first_half_pairs_on();  // FIRST_HALF_PAIRS.get()
std::optional<std::pair<std::uint32_t, std::uint64_t>> test_memory();  // intersect TEST_MEMORY
using PackedLayout = std::vector<std::pair<std::uint8_t, std::vector<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>>>>;
PackedLayout packed_layout(std::span<const SigElement> elements,
                           std::span<const std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs);
std::optional<std::pair<std::uint8_t, std::uint8_t>> packed_half(
    std::span<const SigElement> elements,
    std::span<const std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs, std::uint8_t a,
    std::uint8_t h);
std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> input_packs(
    const StageIo& io, std::span<const SigElement> inputs);
HashMap<std::uint8_t, std::pair<bool, std::optional<std::uint8_t>>> half_outputs(
    const StageIo& io, std::span<const SigElement> outputs);
HashMap<std::uint8_t, std::uint8_t> half_input_partners(const StageIo& io,
                                                        std::span<const SigElement> inputs);
std::optional<std::uint8_t> interp_mode(Barycentric kind);

inline constexpr float TWO_PI = 6.28318530717958647692f;  // std::f32::consts::TAU
// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
inline constexpr std::uint16_t GS_ALLOC_REQ = 9;
// `s_sendmsg` id that ends a pixel wave's ordered section.
inline constexpr std::uint16_t ORDERED_PS_DONE = 7;
// Hardware register (of `s_setreg_b32`) a pixel wave names its packer in.
inline constexpr std::uint16_t POPS_PACKER = 25;
// The operand `src_pops_exiting_wave_id`.
inline constexpr std::uint8_t POPS_EXITING = 239;

// ---- the emitter -------------------------------------------------------------------------------
class Lower {
public:
    Lower(const Parts& x, const HashMap<Value, Why>& bad);
    // Consumes the emitter. Throws StopRetry or ShaderError.
    Lowered run(bool counted);

    // Emitter state.
    Shader s;
    const Parts& x;
    const HashMap<Value, Why>& bad;
    HashMap<Value, Why> lane_fail;
    HashMap<Value, Val> vals;
    HashMap<Value, V> floats;
    // DXIL block per IR block (unreachable: none).
    std::vector<std::optional<std::size_t>> block;
    std::size_t cur = 0;
    HashMap<std::pair<std::size_t, Bool>, V> bools;
    HashMap<Value, V> lane_bits;
    // IR phi location by value.
    HashMap<Value, std::pair<std::size_t, std::size_t>> phi_at;
    // Descriptor dwords with no host value.
    HashSet<Value> stale;
    HashSet<Site> ordered;
    HashSet<std::pair<std::size_t, std::uint32_t>> coherent;
    std::optional<V> root_copy;
    std::optional<V> page_there;
    HashMap<Value, V> mask_phis;
    std::vector<std::tuple<V, std::size_t, std::size_t>> pending_masks;
    std::vector<std::tuple<V, std::size_t, std::size_t>> data_phis;
    HashSet<Value> read_phis;
    HashMap<Value, std::pair<std::size_t, std::uint32_t>> params;
    std::optional<V> user_data;
    BTreeMap<std::uint32_t, UserData> user_reads;
    std::vector<std::array<std::uint32_t, 4>> samplers;
    HashMap<std::pair<Site, bool>, const Source*> descriptors;
    HashMap<Site, Address> scalar;
    HashMap<Site, const Export*> exports;
    HashMap<Site, const Interp*> interps;
    HashMap<std::tuple<std::size_t, Site, std::uint8_t, bool, std::pair<std::uint32_t, std::uint32_t>>, V> handles;
    HashMap<std::tuple<std::size_t, Site, std::uint8_t>, V> strides;
    HashMap<std::pair<std::size_t, std::size_t>, std::size_t> edges;
    std::vector<bool> exported;
    std::vector<std::size_t> rpo;
    std::vector<std::optional<std::size_t>> tail;
    bool writes_uavs = false;
    std::optional<V> src0;
    bool console_lanes = false;
    std::optional<V> flat_;
    HashSet<std::size_t> no_barrier;
    HashSet<std::size_t> fence_only;
    bool split_lds = false;
    std::optional<std::pair<std::vector<bool>, std::vector<std::pair<bool, bool>>>> lds_meet;
    std::optional<std::tuple<std::uint32_t, bool, std::vector<bool>>> exchange;
    bool local = false;
    bool local_ok = false;
    bool local_any = false;
    HashSet<Value> uniform;
    own_bits::OwnBits own_bits;
    std::optional<V> clane;
    std::optional<V> pair;
    std::optional<std::uint32_t> halves;
    std::optional<V> wave_base_;
    HashMap<std::uint8_t, std::pair<bool, std::optional<std::uint8_t>>> half_outputs_;
    MaskData mask_data = MaskData::None;
    bool mask_ok = false;
    HashMap<Value, std::pair<V, V>> ballots;
    HashSet<Value> balloting;
    HashMap<Value, std::size_t> def_block;
    std::size_t cur_ir = 0;
    HashMap<std::size_t, std::unordered_map<std::size_t, wave::Section>> sections;
    HashSet<Value> section_out;
    std::optional<std::tuple<V, std::uint32_t, std::uint32_t>> code_table;
    HashMap<std::size_t, std::pair<V, std::vector<std::pair<V, std::size_t>>>> counters;
    std::optional<std::size_t> abort;

    // Lowering methods.
    V code_dword(V byte);
    void flat(const Inst& inst, std::size_t b, std::size_t i);
    void find_sections();
    void section_inst(std::size_t b, std::size_t i, std::size_t start);
    V shape(wave::Shape shape, bool own);
    V active_f64_sum(Value lo, Value hi, Value mask, bool high);
    V add_halves(V a, V b);
    V half_sum(V x);
    bool exported_out(std::size_t b) const;
    bool has_done_export(std::size_t b) const;
    void inputs();  // (Result<(), Stop>)
    bool pops_collision(Value v) const;
    Val input(Input input);
    Val compute_id(std::string_view name, std::uint32_t opcode, std::uint8_t c);
    std::optional<std::pair<std::size_t, std::uint8_t>> find_input(std::uint8_t row, std::uint8_t channel) const;
    Val load_input(std::size_t id, std::uint8_t col);
    Val load_any_input(std::size_t id, std::uint8_t col);
    Val get(Value v);
    bool lazy_barycentric(Value v);  // (Option<Result<(), Why>>)
    Val barycentric(Barycentric kind, std::uint8_t index);
    V i32_of(Value v);
    Val user(std::size_t param, std::uint32_t dword);
    std::uint32_t user_first() const;
    V host_word(RootType root, std::uint8_t slot, std::uint32_t dword);
    V user_at(std::uint32_t index);
    bool read_as_number(Value v) const;
    bool computed_with(Value v, std::uint32_t depth) const;
    bool is_counter(const Source& src) const;
    std::optional<std::pair<std::size_t, std::uint32_t>> entry(const Source& src) const;
    void find_coherent();
    std::optional<bool> sized_view(const Inst& inst, Site site, std::uint8_t code, std::uint32_t count) const;
    std::optional<std::pair<V, V>> view_address(Site site);
    std::pair<V, V> address_plus(V lo, V hi, std::int32_t off);
    std::pair<V, V> memory(V lo, V hi, std::optional<V> guard, bool store);
    std::optional<std::pair<V, V>> root_address(const Source& src);
    std::optional<std::pair<V, V>> root_memory(const Source& src, V off, std::optional<V> guard, bool store);
    void memory_flat(const Inst& inst, std::size_t b, std::size_t i);
    void memory_smem(const Inst& inst);
    V mask_dword(Value v);
    V operand(const Inst& inst, const Operand& op);
    V f16_operand(const Inst& inst, const Operand& op);
    V as_float(V x);
    V as_int(V f);
    static std::optional<std::pair<std::uint8_t, bool>> sdwa_src(const Inst& inst, std::size_t k);
    V select_bits(V x, std::uint8_t sel, bool sext);
    V fsrc(const Inst& inst, std::size_t k);
    V src(const Inst& inst, std::size_t k);
    V unary(std::uint32_t op, V f);
    V binary(std::uint32_t op, V a, V b);
    V max_min(bool max, V a, V b);
    V fmad(V a, V b, V c);
    V f16_toward_zero(V f);
    V f16_nearest(V f);
    V bfe(V a, V off, V width, bool signed_);
    V is_nan_f32(V f);
    V med3(V a, V b, V third);
    V fract(V a, float threshold, float below_one);
    V trig_turns(V x, bool cosine);
    V mul_hi(V a, V b, bool signed_);
    V legacy_mul(V a, V b);
    V boolean(Bool b, std::size_t at);
    V atom(Atom atom, std::size_t at);
    std::optional<V> wave_base();
    V console_thread();
    V console_lane();
    void group_sync(std::size_t block);
    std::pair<V, V> exchange_halves(std::size_t at, V mine);
    V exchange_pair(std::uint32_t slot);
    V scc_some(Value scc);
    V wave_wide(std::size_t b, Bool when);
    bool all_lanes(std::size_t b) const;
    V lane_bit(std::size_t at, V lo, std::optional<V> hi);
    std::optional<Value> pair_high(Value lo) const;
    std::optional<V> guard(std::size_t b, std::size_t i);
    void set(std::optional<Value> v, Val val);
    void write(const Inst& inst, Reg reg, V x, std::optional<V> guard);
    void write_placed(const Inst& inst, Reg reg, V x, std::optional<V> guard);
    static std::optional<std::pair<std::uint8_t, std::uint8_t>> sdwa_dst(const Inst& inst);
    V place_bits(const Inst& inst, Reg reg, V x, std::uint8_t sel, std::uint8_t unused);
    void write_f(const Inst& inst, V f, std::optional<V> guard);
    void write_i(const Inst& inst, V x, std::optional<V> guard);
    std::optional<std::pair<std::size_t, std::size_t>> begin_guarded(std::optional<V> guard);
    std::vector<V> end_guarded(std::optional<std::pair<std::size_t, std::size_t>> at, std::span<const std::pair<V, V>> values);
    void cross_lane(bool quad);
    Val thread_id(std::uint8_t c);
    void mesh_prologue(Mesh mesh);
    V mesh_thread();
    std::uint32_t wave() const;
    V lane_index();
    V quad_pick(V x, V src);
    bool helpers_read(bool quad) const;
    V quad_read(V x, std::array<std::uint8_t, 4> lanes);
    std::pair<V, V> first_lane(V x, std::optional<V> guard);
    V helper_lane();
    std::pair<V, V> first_live_lane(V x, std::optional<V> guard);
    V read_lane(V x, V lane);
    V float_of(Value v);
    std::vector<V> vgprs(const Inst& inst, const Operand& op, std::uint8_t n);
    void ds(const Inst& inst, std::size_t b, std::size_t i);
    V lds_float_min_max(V p, V data, bool max);
    V lds_float_pick(V mem, V data, bool max);
    void inst(std::size_t b, std::size_t i);  // (Result<(), Stop>)
    void memory_fence();
    void lds_meeting();
    bool dead(Value v, std::vector<Value>& seen) const;
    bool uniform_test(const Inst& inst) const;
    bool is_mask(std::optional<Value> v) const;
    void sop(const Inst& inst);
    V quad_mask(V pixels);
    std::pair<V, V> add_sub_carry(std::size_t c, bool add, V a, V b, std::optional<V> cin);
    V scc_in(const Inst& inst);
    bool within_exec(Value scc) const;
    static bool wide_loads();
    std::vector<V> raw_dwords(V h, V addr, std::uint32_t count);
    void smem(const Inst& inst, Site site);
    void descriptor_data(const Inst& inst, Site site);
    std::optional<std::tuple<std::uint8_t, std::pair<std::uint32_t, std::uint32_t>, std::optional<std::uint32_t>, bool, const UsedResource*>> unused_descriptor_kind(const Inst& inst, Site site, std::uint8_t code, std::uint32_t count) const;
    void code_smem(const Inst& inst, Address a);
    V cbuffer_dword(V h, V a);
    std::vector<V> format_load(V h, V off, std::uint8_t format, std::uint32_t channels);
    V norm_quotient(V a, std::uint32_t w);
    V exact_norm(V a, std::uint32_t w);
    std::vector<V> d16_channels(std::uint8_t format, std::vector<V> channels);
    std::pair<std::uint32_t, std::optional<V>> built_offset(const Source& src);
    std::optional<V> root_bias(std::size_t param);
    const UsedResource* used_kind(const Source& src) const;
    V handle(Site site, const Source& src, bool sampler, std::pair<std::uint32_t, std::uint32_t> props);
    std::pair<V, bool> raw_twin(const Source& src, V index, bool sampler, std::pair<std::uint32_t, std::uint32_t> props);
    V descriptor_phi(std::size_t block, std::span<const Source> args, bool sampler, std::uint32_t depth);
    std::uint32_t register_sampler(std::span<const std::uint32_t> words, bool cmp);
    std::pair<std::optional<V>, std::vector<std::uint32_t>> sampler_phi_choice(std::size_t block, std::span<const Source> args, bool cmp, std::uint32_t depth);
    void collect_constant_samplers(std::size_t block, std::span<const Source> args, std::uint32_t depth, std::vector<std::array<std::uint32_t, 4>>& distinct) const;
    V sampler_index_phi(std::size_t block, std::span<const Source> args, std::uint32_t lower, std::span<const std::array<std::uint32_t, 4>> distinct, std::uint32_t depth);
    std::optional<V> slot_stride(const Source& src);
    std::optional<std::pair<V, V>> integer_view(const Source& src);
    std::optional<V> slot_word(const Source& src);
    V lane_heap_index(const Source& src, bool sampler);
    V fetched_index(Site load, std::uint8_t first, bool sampler);
    V heap_index(Site load, std::uint8_t first, bool sampler);
    void valu(const Inst& inst, std::size_t b, std::size_t i);
    std::pair<V, std::optional<V>> dpp(const Inst& inst, const Dpp& dpp);
    void valu_op(const Inst& inst, std::optional<V> guard);
    V src64(const Inst& inst, std::size_t k);
    std::pair<V, V> f64_pair(const Inst& inst, std::size_t k);
    V f64_to_f32_bits(V lo, V hi);
    V join64(V lo, V hi);
    V f64_bits_to_f32_bits(V lo, V hi);
    std::pair<V, V> split64(V x);
    V half_src(const Inst& inst, std::size_t k);
    V half_fsrc(const Inst& inst, std::size_t k);
    void write_half(const Inst& inst, V x, std::optional<V> guard);
    bool valu16(const Inst& inst, std::string_view name, std::optional<V> guard);  // (Option<Result<(), Why>>)
    void valu16_op(const Inst& inst, std::string_view name, std::optional<V> guard, std::span<const std::string_view> float1, std::span<const std::string_view> float2, std::span<const std::string_view> float3);
    V cvt_norm16(const Inst& inst, bool signed_);
    V packed_src(const Inst& inst, std::size_t k, bool high, bool float_);
    V fmac_half(const Inst& inst, std::size_t k, bool high);
    void packed16(const Inst& inst, std::string_view op, std::optional<V> guard);
    V integer(const Inst& inst, Why name);  // (name: &'static str)
    void compare(const Inst& inst, std::string_view rest);
    void lane(const Inst& inst, std::optional<bool> value);
    void lane_bits_for(const Inst& inst, V r);
    void interp(const Inst& inst, Site site);
    bool per_vertex(std::uint8_t row) const;
    bool interpolated(std::uint8_t row) const;
    V eval_channel(std::uint8_t row, std::uint8_t ch, Eval eval);
    V vertex_input(std::size_t id, std::uint8_t col, std::uint8_t k);
    V low_half(const Inst& inst, std::uint8_t reg, V x);
    void image(const Inst& inst, std::size_t b, std::size_t i);
    std::array<V, 3> cube_direction(V s, V t, V face);
    void buffer(const Inst& inst, std::size_t b, std::size_t i);
    void export_(std::size_t b, std::size_t i);
    void emit_indices(const Export& e, std::optional<V> guard);
    std::optional<std::uint8_t> unnamed_row() const;
    void exit(std::size_t b);  // (Result<(), Stop>)
    void enter_loop(std::size_t from, std::size_t to, std::size_t via);
    V count_iteration(std::size_t to, std::uint8_t p);
    std::size_t abort_block();
    std::size_t edge(std::size_t from, std::size_t to);
    void fill_phis();  // (Result<(), Stop>)

    // Ray intersection methods (lower_intersect.cpp).
    void bvh(const Inst& inst, std::size_t b, std::size_t i);  // irs:64
    std::tuple<V, V, V> bvh_memory(V low, V high, std::optional<V> guard);  // irs:230
    NodeAt bvh_second_half(NodeAt at);  // irs:256
    std::vector<V> bvh_node_words(NodeAt at, std::optional<V> base, std::uint32_t offset, std::uint32_t count);  // irs:280
    std::pair<V, V> bvh_address(V word0, V word1, V pointer, std::optional<V> pointer_high);  // irs:316
    std::array<V, 4> bvh_boxes(NodeAt at, V kind, const Ray& ray, bool sort);  // irs:335
    std::array<V, 4> bvh_triangle(NodeAt at, V kind, const Ray& ray);  // irs:467
};

}  // namespace x2w::vs::lower
