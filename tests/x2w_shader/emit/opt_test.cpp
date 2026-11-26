// Port of the tests in the upstream Rust module emit/bitcode/opt.rs (#[cfg(test)] mod tests).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/opt.h"
#include "test_util.h"

using namespace x2w::vs::bitcode;
using namespace x2w::vs::bitcode::opt;

namespace {

using Stores = std::vector<std::vector<V>>;

// The FNV-1a of each module a test writes, printed to compare with the Rust tests' modules.
const char* current_test = "";

// What the Rust tests' modules hash to (cargo test of the upstream shader crate with opt.rs's tests
// printing them), in the order the tests below write them.
constexpr std::uint64_t RUST_WRITTEN[] = {
    0xf684d713293b47f7ull, 0xb05de7565dfe5bdaull, 0x625329b87e8b0c77ull, 0xd741149b576b7948ull,
    0x542e0847305d4c96ull, 0xd0f74cd9b590bd05ull, 0x5d54e661dd1f406dull, 0xe98644c9c0828d03ull,
    0x9569fe82dd5dadc4ull, 0x99daecdb13de1ba1ull, 0xd3a8b267281a6ae0ull,
    // mixed_cross_check (added here, see there).
    0x81618006d371acd7ull,
};
std::size_t written_count = 0;

void write_and_note(const Module& m) {
    std::vector<std::uint8_t> bytes = m.write();
    const std::uint64_t h = test_util::fnv(bytes);
    std::printf("%s: written %016llx\n", current_test, (unsigned long long)h);
    CHECK(written_count < std::size(RUST_WRITTEN) && RUST_WRITTEN[written_count] == h);
    ++written_count;
}

std::uint64_t mask(std::uint32_t bits) { return bits >= 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1; }

std::int64_t signed_(std::uint32_t bits, std::uint64_t x) {
    if (bits < 64 && ((x >> (bits - 1)) & 1) != 0) return std::int64_t(x | ~mask(bits));
    return std::int64_t(x);
}

std::optional<std::uint32_t> int_bits(const Module& m, Ty ty) {
    const TypeDef& d = m.types.at(ty.id);
    if (d.kind == TypeDef::Int) return d.n;
    return std::nullopt;
}

// `main` with one block and a function of each kind of effect.
struct Kit {
    Module m;
    std::size_t b = 0;
    V pure, reads, writes;
};

Kit kit() {
    Kit k;
    Module& m = k.m;
    Ty i32t = m.int_(32);
    Ty voidt = m.void_();
    Ty fty = m.func_ty(i32t, {i32t, i32t});
    m.ptr(fty);
    k.pure = m.function("dx.op.binary.i32", fty, Attr::ReadNone, false);
    k.reads = m.function("dx.op.bufferLoad.i32", fty, Attr::ReadOnly, false);
    k.writes = m.function("dx.op.bufferStore.i32", fty, Attr::NoUnwind, false);
    Ty main_ty = m.func_ty(voidt, {});
    m.ptr(main_ty);
    m.function("main", main_ty, std::nullopt, true);
    k.b = m.new_block();
    return k;
}

std::size_t body(const Module& m, std::size_t b) { return m.blocks[b].body.size(); }

// The operands of each store left in block `b`.
Stores stores(const Module& m, std::size_t b) {
    Stores r;
    for (std::uint32_t i : m.blocks[b].body) {
        const Op& op = m.insts[i].op;
        if (op.kind == Op::Call && m.funcs[op.index].name.find("Store") != std::string::npos) r.push_back(op.list);
    }
    return r;
}

void unread_results_without_effect_are_dropped() {
    Kit k = kit();
    Module& m = k.m;
    std::size_t b = k.b;
    V one = m.i32(1);
    V two = m.i32(2);
    V used = m.call(b, k.reads, {one, two});
    V dead = m.call(b, k.reads, {two, one});
    V chain = m.bin(b, Bin::Add, dead, one);
    (void)m.call(b, k.pure, {chain, two});
    // A store's result is unread and stays, with what it reads.
    V kept = m.bin(b, Bin::Add, used, two);
    m.call(b, k.writes, {kept, one});
    m.terminate(b, Term::ret());
    m.optimise(DEAD);
    CHECK(body(m, b) == 3);  // the load, the add and the store
    write_and_note(m);
}

void branch_conditions_and_phi_cycles() {
    Kit k = kit();
    Module& m = k.m;
    std::size_t b = k.b;
    Ty i32t = m.int_(32);
    V zero = m.i32(0);
    V one = m.i32(1);
    std::size_t head = m.new_block();
    std::size_t exit = m.new_block();
    m.terminate(b, Term::br(head));
    // A counter that decides the branch, and a register that only feeds itself round the loop.
    V count = m.phi(head, i32t);
    V idle = m.phi(head, i32t);
    V next = m.bin(head, Bin::Add, count, one);
    V idle_next = m.call(head, k.reads, {idle, one});
    V more = m.cmp(head, pred::IULT, next, one);
    m.terminate(head, Term::cond_br(more, head, exit));
    m.set_phi(count, {{zero, b}, {next, head}});
    m.set_phi(idle, {{zero, b}, {idle_next, head}});
    m.call(exit, k.writes, {zero, one});
    m.terminate(exit, Term::ret());
    m.optimise(DEAD);
    CHECK(m.blocks[head].phis.size() == 1);  // the counter
    CHECK(body(m, head) == 2);               // the step and the test
    CHECK(body(m, exit) == 1);
    write_and_note(m);
}

void wave_operations_have_no_effect_of_their_own() {
    Kit k = kit();
    Ty i32t = k.m.int_(32);
    Ty fty = k.m.func_ty(i32t, {i32t, i32t});
    V first = k.m.function("dx.op.waveReadLaneFirst.i32", fty, Attr::NoUnwind, false);
    V barrier = k.m.function("dx.op.barrier", fty, Attr::NoDuplicate, false);
    V one = k.m.i32(1);
    k.m.call(k.b, first, {one, one});
    k.m.call(k.b, barrier, {one, one});
    k.m.terminate(k.b, Term::ret());
    k.m.optimise(DEAD);
    CHECK(body(k.m, k.b) == 1);  // the barrier
    // The declarations nothing calls any more are gone, and the call that stayed names its
    // function by the new number.
    std::vector<std::string> names;
    for (const Func& f : k.m.funcs) names.push_back(f.name);
    CHECK((names == std::vector<std::string>{"main", "dx.op.barrier"}));
    const Op& op = k.m.insts[k.m.blocks[k.b].body[0]].op;
    CHECK(op.kind == Op::Call);
    if (op.kind == Op::Call) CHECK(k.m.funcs[op.index].name == "dx.op.barrier");
    write_and_note(k.m);
}

void constants_and_identities_fold() {
    Kit kt = kit();
    Module& m = kt.m;
    std::size_t b = kt.b;
    Ty i32t = m.int_(32);
    Ty f32t = m.float_();
    V zero = m.i32(0);
    V one = m.i32(1);
    V all = m.i32(0xFFFFFFFFu);
    V x = m.call(b, kt.reads, {one, one});
    // x + 0, -1 & x, x >> 0, x * 1, x | x: all x.
    V a = m.bin(b, Bin::Add, x, zero);
    a = m.bin(b, Bin::And, all, a);
    a = m.bin(b, Bin::LShr, a, zero);
    a = m.bin(b, Bin::Mul, a, one);
    a = m.bin(b, Bin::Or, a, a);
    // There and back.
    V f = m.cast(b, Cast::Bitcast, a, f32t);
    a = m.cast(b, Cast::Bitcast, f, i32t);
    // Constants: (262168 >> 16) & 127 = 4, 7 & 31 = 7, -8 >>s 1 = -4.
    V k = m.i32(262168);
    V sixteen = m.i32(16);
    k = m.bin(b, Bin::LShr, k, sixteen);
    V k127 = m.i32(127);
    k = m.bin(b, Bin::And, k, k127);
    V minus8 = m.i32(std::uint32_t(-8));
    V s = m.bin(b, Bin::AShr, minus8, one);
    // A shift by the width is left alone.
    V k32 = m.i32(32);
    V wide = m.bin(b, Bin::Shl, one, k32);
    // A select between a value and itself, and on a constant.
    V c = m.cmp(b, pred::IULT, zero, one);
    V sel = m.select(b, c, a, wide);
    V same = m.select(b, c, sel, sel);
    m.call(b, kt.writes, {same, k});
    m.call(b, kt.writes, {s, wide});
    m.terminate(b, Term::ret());
    m.optimise(FOLD | DEAD);
    CHECK(body(m, b) == 4);  // the read, the wide shift and two stores
    V four = m.i32(4);
    V minus4 = m.i32(std::uint32_t(-4));
    CHECK((stores(m, b) == Stores{{x, four}, {minus4, wide}}));
    write_and_note(m);
}

void float_arithmetic_is_not_folded() {
    Kit kt = kit();
    Module& m = kt.m;
    std::size_t b = kt.b;
    Ty f32t = m.float_();
    Ty fty = m.func_ty(f32t, {f32t, f32t});
    V input = m.function("dx.op.loadInput.f32", fty, Attr::ReadNone, false);
    V zero = m.f32(0);
    V one = m.f32(std::bit_cast<std::uint32_t>(1.0f));
    V two = m.f32(std::bit_cast<std::uint32_t>(2.0f));
    V x = m.call(b, input, {zero, zero});
    V y = m.call(b, input, {one, one});
    V a = m.bin(b, Bin::Add, x, zero);
    V p = m.bin(b, Bin::Mul, a, one);
    // The same sum the other way round is another instruction.
    V xy = m.bin(b, Bin::Add, x, y);
    V yx = m.bin(b, Bin::Add, y, x);
    // Constants compare here: 1 < 2, and nothing equals a NaN.
    V nan = m.f32(0x7FC00000u);  // f32::NAN.to_bits()
    V lt = m.cmp(b, pred::OLT, one, two);
    V eq = m.cmp(b, pred::OEQ, nan, nan);
    V une = m.cmp(b, pred::UNE, nan, one);
    V sel = m.select(b, lt, xy, yx);
    sel = m.select(b, eq, p, sel);
    sel = m.select(b, une, sel, p);
    m.call(b, kt.writes, {sel, p});
    m.terminate(b, Term::ret());
    m.optimise(ALL);
    // Two inputs, x + 0.0, * 1.0, x + y and the store: y + x and the selects are gone.
    CHECK(body(m, b) == 6);
    CHECK((stores(m, b) == Stores{{xy, p}}));
    write_and_note(m);
}

// entry: `c = x < y`, branch to `then` or `other`, which join.
struct Diamond {
    Kit k;
    V x, y, c;
    std::size_t then = 0, other = 0, join = 0;
};

Diamond diamond() {
    Diamond d;
    d.k = kit();
    Kit& k = d.k;
    V one = k.m.i32(1);
    V two = k.m.i32(2);
    d.x = k.m.call(k.b, k.reads, {one, one});
    d.y = k.m.call(k.b, k.reads, {two, two});
    d.c = k.m.cmp(k.b, pred::IULT, d.x, d.y);
    d.then = k.m.new_block();
    d.other = k.m.new_block();
    d.join = k.m.new_block();
    k.m.terminate(k.b, Term::cond_br(d.c, d.then, d.other));
    k.m.terminate(d.then, Term::br(d.join));
    k.m.terminate(d.other, Term::br(d.join));
    k.m.terminate(d.join, Term::ret());
    return d;
}

void a_value_is_reused_below_its_block_and_not_beside_it() {
    Diamond d = diamond();
    Module& m = d.k.m;
    std::size_t b = d.k.b;
    V x = d.x, y = d.y;
    // The entry's value again in a branch; a branch's value again in the other branch and in the
    // join (neither is on every path).
    V top = m.call(b, d.k.pure, {x, y});
    m.call(b, d.k.writes, {top, top});
    V again = m.call(d.then, d.k.pure, {x, y});
    V side = m.bin(d.then, Bin::Add, y, x);
    m.call(d.then, d.k.writes, {again, side});
    V beside = m.bin(d.other, Bin::Add, x, y);
    m.call(d.other, d.k.writes, {beside, beside});
    V after = m.bin(d.join, Bin::Add, x, y);
    m.call(d.join, d.k.writes, {after, after});
    m.optimise(REUSE);
    CHECK((stores(m, d.then) == Stores{{top, side}}));
    CHECK(body(m, d.then) == 2);
    CHECK(body(m, d.other) == 2);
    CHECK(body(m, d.join) == 2);
    write_and_note(m);
}

void a_branch_taken_says_what_its_condition_is() {
    {
        Diamond dm = diamond();
        Module& m = dm.k.m;
        std::size_t b = dm.k.b;
        V x = dm.x, y = dm.y, c = dm.c;
        V three = m.i32(3);
        V d = m.call(b, dm.k.reads, {three, three});
        d = m.cmp(b, pred::NE, d, three);
        V both = m.bin(b, Bin::And, c, d);
        // Behind the branch: the condition again (a guard), its opposite comparison, and a
        // guarded write.
        V guard = m.bin(dm.then, Bin::And, c, d);
        V old = m.select(dm.then, guard, x, y);
        V not_ = m.cmp(dm.then, pred::IUGE, x, y);
        V nw = m.select(dm.then, not_, y, old);
        m.call(dm.then, dm.k.writes, {nw, nw});
        V kept = m.select(dm.other, c, x, y);
        m.call(dm.other, dm.k.writes, {kept, kept});
        // After the join nothing is known.
        V open = m.select(dm.join, c, x, y);
        m.call(dm.join, dm.k.writes, {open, open});
        m.optimise(FOLD | FACTS);
        // then: select(d, x, y) and the store; other: y is stored.
        CHECK(body(m, dm.then) == 2);
        Stores st = stores(m, dm.then);
        CHECK(!st.empty() && st[0][0].kind == V::Inst);
        if (!st.empty() && st[0][0].kind == V::Inst) {
            const Op& op = m.insts[st[0][0].i].op;
            CHECK(op.kind == Op::Select && op.a == d && op.b == x && op.c == y);
        }
        CHECK((stores(m, dm.other) == Stores{{y, y}}));
        CHECK(body(m, dm.join) == 2);
        (void)both;
        write_and_note(m);
    }
    // Both terms of an `and` that held hold behind it.
    {
        Diamond dm = diamond();
        Module& m = dm.k.m;
        std::size_t b = dm.k.b;
        V x = dm.x, y = dm.y, c = dm.c;
        V three = m.i32(3);
        V d = m.call(b, dm.k.reads, {three, three});
        d = m.cmp(b, pred::NE, d, three);
        V both = m.bin(b, Bin::And, c, d);
        std::size_t inner = m.new_block();
        std::size_t out = m.new_block();
        m.terminate(b, Term::cond_br(both, inner, out));
        V a = m.select(inner, c, x, y);
        V z = m.select(inner, d, a, y);
        m.call(inner, dm.k.writes, {z, z});
        m.terminate(inner, Term::br(out));
        // Where the `and` did not hold either term may.
        V w = m.select(out, d, x, y);
        m.call(out, dm.k.writes, {w, w});
        m.terminate(out, Term::ret());
        m.optimise(FOLD | FACTS);
        CHECK((stores(m, inner) == Stores{{x, x}}));
        CHECK(body(m, out) == 2);
        write_and_note(m);
    }
}

void a_branch_on_a_constant_goes_one_way() {
    Diamond dm = diamond();
    Module& m = dm.k.m;
    V x = dm.x, y = dm.y, c = dm.c;
    Ty i32t = m.int_(32);
    // Behind the branch the same test again: its other side is never run, and the join behind it
    // has one value.
    std::size_t deep = m.new_block();
    std::size_t never = m.new_block();
    std::size_t meet = m.new_block();
    m.terminate(dm.then, Term::cond_br(c, deep, never));
    m.terminate(deep, Term::br(meet));
    m.call(never, dm.k.writes, {y, y});
    m.terminate(never, Term::br(meet));
    V p = m.phi(meet, i32t);
    m.set_phi(p, {{x, deep}, {y, never}});
    m.call(meet, dm.k.writes, {p, p});
    m.terminate(meet, Term::br(dm.join));
    V q = m.phi(dm.join, i32t);
    m.set_phi(q, {{p, meet}, {y, dm.other}});
    m.call(dm.join, dm.k.writes, {q, q});
    m.optimise(ALL);
    const auto& t = m.blocks[dm.then].term;
    CHECK(t && t->kind == Term::Br && t->t == deep);
    CHECK((stores(m, meet) == Stores{{x, x}}));
    CHECK(m.blocks[meet].phis.empty());
    CHECK(body(m, never) == 0);  // emptied: nothing reaches it
    CHECK(!m.blocks[dm.join].phis.empty());
    if (!m.blocks[dm.join].phis.empty()) {
        const Op& op = m.insts[m.blocks[dm.join].phis[0]].op;
        CHECK(op.kind == Op::Phi);
        std::vector<std::pair<V, std::size_t>> want = {{x, meet}, {y, dm.other}};
        CHECK(op.incoming == want);
    }
    write_and_note(m);
}

void a_read_is_reused_within_its_block_until_a_store() {
    Diamond dm = diamond();
    Module& m = dm.k.m;
    std::size_t b = dm.k.b;
    V x = dm.x, y = dm.y;
    std::size_t then = dm.then;
    Ty i32t = m.int_(32);
    // The entry reads (1, 1) for x: again in the entry it is x; in the next block it is read
    // again.
    V one = m.i32(1);
    V again = m.call(b, dm.k.reads, {one, one});
    m.call(b, dm.k.writes, {again, y});
    V after = m.call(b, dm.k.reads, {one, one});
    V twice = m.call(b, dm.k.reads, {one, one});
    V sum = m.bin(b, Bin::Add, after, twice);
    m.call(b, dm.k.writes, {sum, sum});
    V below = m.call(then, dm.k.reads, {one, one});
    m.call(then, dm.k.writes, {below, below});
    // A constant buffer cannot change: one load for the function.
    Ty cfty = m.func_ty(i32t, {i32t, i32t});
    V cb = m.function("dx.op.cbufferLoadLegacy.i32", cfty, Attr::ReadOnly, false);
    V top = m.call(b, cb, {one, one});
    m.call(b, dm.k.writes, {top, top});
    V under = m.call(then, cb, {one, one});
    m.call(then, dm.k.writes, {under, under});
    m.optimise(REUSE | LOADS);
    Stores st = stores(m, b);
    CHECK(st.size() == 3);
    if (st.size() == 3) {
        CHECK((st[0] == std::vector<V>{x, y}));  // the second read is the first
        CHECK(st[1][0].kind == V::Inst);
        if (st[1][0].kind == V::Inst) {
            // Read again after the store, once.
            const Op& op = m.insts[st[1][0].i].op;
            CHECK(op.kind == Op::Bin && op.code == std::uint8_t(Bin::Add) && op.a == after && op.b == after);
        }
        CHECK((st[2] == std::vector<V>{top, top}));
    }
    CHECK((stores(m, then) == Stores{{below, below}, {top, top}}));
    write_and_note(m);
}

void a_join_of_one_value_and_two_equal_joins() {
    Diamond dm = diamond();
    Module& m = dm.k.m;
    V x = dm.x, y = dm.y;
    Ty i32t = m.int_(32);
    V one = m.phi(dm.join, i32t);
    m.set_phi(one, {{x, dm.then}, {x, dm.other}});
    V p = m.phi(dm.join, i32t);
    m.set_phi(p, {{x, dm.then}, {y, dm.other}});
    V q = m.phi(dm.join, i32t);
    m.set_phi(q, {{y, dm.other}, {x, dm.then}});
    m.call(dm.join, dm.k.writes, {one, p});
    V sum = m.bin(dm.join, Bin::Add, p, q);
    m.call(dm.join, dm.k.writes, {sum, q});
    m.optimise(FOLD | REUSE);
    CHECK((m.blocks[dm.join].phis == std::vector<std::uint32_t>{p.i}));
    Stores st = stores(m, dm.join);
    CHECK(st.size() == 2);
    if (st.size() == 2) {
        CHECK((st[0] == std::vector<V>{x, p}));
        CHECK(st[1][1] == p);
    }
    write_and_note(m);
}

// Not in opt.rs: a module through the paths the tests above do not reach (constant tables, shared
// memory, loads, stores, atomics, i64 and i8 folds, metadata, bounded, float_k, usage), with the
// values the same module gives in Rust (the identical test added to a scratch copy of opt.rs).
std::string bits_line(const std::vector<std::optional<float>>& fs) {
    std::string line;
    char buf[16];
    for (const auto& f : fs) {
        if (f) {
            std::snprintf(buf, sizeof buf, " %08x", std::bit_cast<std::uint32_t>(*f));
            line += buf;
        } else {
            line += " none";
        }
    }
    return line;
}

std::string usage_line(const Usage& u) {
    std::string s;
    for (bool x : {u.wave_ops, u.rovs, u.int64, u.typed_uav_loads, u.heap_views, u.heap_samplers, u.atomic64})
        s += x ? '1' : '0';
    return s;
}

void mixed_cross_check() {
    Kit kt = kit();
    std::size_t b = kt.b;
    Module& m = kt.m;
    Ty i32t = m.int_(32);
    Ty i64t = m.int_(64);
    Ty i8t = m.int_(8);
    Ty f32t = m.float_();
    Ty arr = m.array(4, i32t);
    V one = m.i32(1);
    V two = m.i32(2);
    V three = m.i32(3);
    V table_init = m.aggregate(arr, {one, two, three, one});
    V table = m.constant_global("table", table_init);
    V unused_init = m.aggregate(arr, {two, two, two, two});
    (void)m.constant_global("unused", unused_init);
    V shared = m.global("shared", arr, 3);
    V x = m.call(b, kt.reads, {one, two});
    V e = m.element(b, table, x);
    V l1 = m.load(b, e);
    V e2 = m.element(b, table, x);
    V l2 = m.load(b, e2);
    V s = m.element(b, shared, one);
    V sl = m.load(b, s);
    m.store(b, s, l1);
    V sl2 = m.load(b, s);
    V at = m.atomic(b, Rmw::Add, s, l2);
    V cx = m.cmpxchg(b, s, at, sl);
    V k64a = m.int_const(i64t, -5);
    V k64b = m.int_const(i64t, 3);
    V q = m.bin(b, Bin::UDiv, k64a, k64b);
    V r = m.bin(b, Bin::URem, k64a, k64b);
    V k8 = m.i8(200);
    V sx = m.cast(b, Cast::SExt, k8, i32t);
    V tr = m.cast(b, Cast::Trunc, sx, i8t);
    V zx = m.cast(b, Cast::ZExt, tr, i32t);
    V w = m.cast(b, Cast::Trunc, q, i32t);
    V w2 = m.cast(b, Cast::Trunc, r, i32t);
    V z64 = m.cast(b, Cast::ZExt, x, i64t);
    V t32 = m.cast(b, Cast::Trunc, z64, i32t);
    V sum = m.bin(b, Bin::Add, sl2, cx);
    sum = m.bin(b, Bin::Add, sum, zx);
    sum = m.bin(b, Bin::Add, sum, w);
    sum = m.bin(b, Bin::Add, sum, w2);
    sum = m.bin(b, Bin::Add, sum, t32);
    m.call(b, kt.writes, {sum, sl});
    Ty fty1 = m.func_ty(f32t, {i32t, f32t});
    Ty fty2 = m.func_ty(f32t, {i32t, f32t, f32t});
    Ty fty3 = m.func_ty(f32t, {i32t, f32t, f32t, f32t});
    V unary = m.function("dx.op.unary.f32", fty1, Attr::ReadNone, false);
    V binary = m.function("dx.op.binary.f32", fty2, Attr::ReadNone, false);
    V tertiary = m.function("dx.op.tertiary.f32", fty3, Attr::ReadNone, false);
    Ty wty = m.func_ty(i32t, {i32t, i32t});
    V wave = m.function("dx.op.waveActiveOp.i32", wty, Attr::NoUnwind, false);
    V half = m.f32(std::bit_cast<std::uint32_t>(0.5f));
    V big = m.f32(std::bit_cast<std::uint32_t>(1000.0f));
    V fx = m.cast(b, Cast::UiToFp, x, f32t);
    V c6 = m.i32(6);
    V c7 = m.i32(7);
    V c13 = m.i32(13);
    V c35 = m.i32(35);
    V c36 = m.i32(36);
    V c46 = m.i32(46);
    V sat = m.call(b, unary, {c7, fx});
    V ab = m.call(b, unary, {c6, fx});
    V sin = m.call(b, unary, {c13, ab});
    V mx = m.call(b, binary, {c35, fx, half});
    V mn = m.call(b, binary, {c36, mx, big});
    V mad = m.call(b, tertiary, {c46, fx, half, big});
    V mul = m.bin(b, Bin::Mul, mad, sat);
    V add = m.bin(b, Bin::Add, mul, mn);
    V bi = m.cast(b, Cast::Bitcast, sin, i32t);
    V bf = m.cast(b, Cast::Bitcast, bi, f32t);
    V ldf = m.cast(b, Cast::Bitcast, x, f32t);
    V mx2 = m.call(b, binary, {c35, ldf, half});
    V mn2 = m.call(b, binary, {c36, mx2, big});
    V kf = m.cast(b, Cast::Bitcast, three, f32t);
    V sel = m.select(b, sl, add, mn2);
    V wv = m.call(b, wave, {x, one});
    V out = m.cast(b, Cast::Bitcast, sel, i32t);
    V out2 = m.cast(b, Cast::Bitcast, bf, i32t);
    m.call(b, kt.writes, {out, out2});
    m.call(b, kt.writes, {wv, wv});
    Md mv1 = m.md_value(table);
    Md mv2 = m.md_value(kt.pure);
    Md node = m.md_node({mv1, mv2});
    m.named_md("cross", {node});
    m.terminate(b, Term::ret());
    std::vector<std::optional<float>> bounds;
    for (V v : {sat, ab, sin, mx, mn, mad, mul, add, bf, mn2, fx, ldf, kf, sel, half, big}) bounds.push_back(m.bounded(v, 8));
    std::string line = bits_line(bounds);
    std::printf("mixed bounded:%s\n", line.c_str());
    CHECK(line == " 00000000 42000000 00000000 42000000 42000000 42000000 42000000 42040000 00000000 41200000 "
                  "42000000 none c2c60000 42040000 bf800000 41200000");
    std::vector<std::optional<float>> ks;
    for (V v : {half, big, kf, ldf, three}) ks.push_back(m.float_k(v));
    line = bits_line(ks);
    std::printf("mixed float_k:%s\n", line.c_str());
    CHECK(line == " 3f000000 447a0000 00000003 none none");
    CHECK(usage_line(m.usage()) == "1011000");
    m.optimise(ALL);
    CHECK(usage_line(m.usage()) == "1001000");
    CHECK(m.globals.size() == 2);
    CHECK(m.funcs.size() == 8);
    CHECK(body(m, b) == 34);
    write_and_note(m);
}

// ---- the passes against an interpreter, on random programs ----

using Trace = std::vector<std::pair<std::string, std::vector<std::uint64_t>>>;

// Runs `m` on `inputs`: the stores it makes, in order (and how it ended). Reads a value before its
// definition ran: a panic.
Trace run(const Module& m, const std::vector<std::uint32_t>& inputs) {
    std::vector<std::optional<std::uint64_t>> vals(m.insts.size());
    std::uint64_t memory[16] = {};
    Trace trace;
    auto eval = [&](V v) -> std::uint64_t {
        switch (v.kind) {
        case V::Inst: {
            const auto& x = vals.at(v.i);
            if (!x) bug("a value defined before its use");
            return *x;
        }
        case V::Const: {
            const auto& [ty, c] = m.consts.at(v.i);
            auto bits = int_bits(m, ty);
            if (c.kind == Const::Int && bits) return std::uint64_t(c.value) & mask(*bits);
            if (c.kind == Const::Float) return std::uint64_t(c.bits);
            return 0;
        }
        default:
            bug("a global or function as a value");
        }
    };
    std::size_t b = 0, from = ~std::size_t(0);
    for (int step = 0; step < 4000; ++step) {
        const Block& blk = m.blocks.at(b);
        // Joins take their values together, before any is assigned.
        std::vector<std::pair<std::uint32_t, std::uint64_t>> joined;
        for (std::uint32_t p : blk.phis) {
            const Op& op = m.insts[p].op;
            if (op.kind != Op::Phi) bug("a join");
            auto it = std::find_if(op.incoming.begin(), op.incoming.end(), [&](const auto& x) { return x.second == from; });
            if (it == op.incoming.end()) bug("a value for the edge taken");
            joined.emplace_back(p, eval(it->first));
        }
        for (const auto& [p, x] : joined) vals[p] = x;
        for (std::uint32_t i : blk.body) {
            const Inst& inst = m.insts[i];
            const auto bits = int_bits(m, inst.ty);
            auto fl = [](std::uint64_t x) { return std::bit_cast<float>(std::uint32_t(x)); };
            std::uint64_t r = 0;
            const Op& op = inst.op;
            switch (op.kind) {
            case Op::Bin: {
                std::uint64_t x = eval(op.a);
                std::uint64_t y = eval(op.b);
                if (bits) {
                    std::uint64_t z;
                    switch (Bin(op.code)) {
                    case Bin::Add:
                        z = x + y;
                        break;
                    case Bin::Sub:
                        z = x - y;
                        break;
                    case Bin::Mul:
                        z = x * y;
                        break;
                    case Bin::And:
                        z = x & y;
                        break;
                    case Bin::Or:
                        z = x | y;
                        break;
                    case Bin::Xor:
                        z = x ^ y;
                        break;
                    case Bin::Shl:
                        if (y >= 64) bug("shift overflow");
                        z = x << y;
                        break;
                    case Bin::LShr:
                        if (y >= 64) bug("shift overflow");
                        z = x >> y;
                        break;
                    case Bin::AShr:
                        if (y >= 64) bug("shift overflow");
                        z = std::uint64_t(signed_(*bits, x) >> y);
                        break;
                    default:
                        bug("no divisions here");
                    }
                    r = z & mask(*bits);
                } else {
                    float fx = fl(x), fy = fl(y);
                    float z;
                    switch (Bin(op.code)) {
                    case Bin::Add:
                        z = fx + fy;
                        break;
                    case Bin::Sub:
                        z = fx - fy;
                        break;
                    default:
                        z = fx * fy;
                        break;
                    }
                    r = std::uint64_t(std::bit_cast<std::uint32_t>(z));
                }
                break;
            }
            case Op::Cmp: {
                const auto wide = int_bits(m, m.type_of(op.a));
                std::uint64_t x = eval(op.a);
                std::uint64_t y = eval(op.b);
                const std::uint8_t p = op.code;
                bool holds;
                if (wide) {
                    std::int64_t sx = signed_(*wide, x), sy = signed_(*wide, y);
                    switch (p) {
                    case pred::EQ:
                        holds = x == y;
                        break;
                    case pred::NE:
                        holds = x != y;
                        break;
                    case pred::IUGT:
                        holds = x > y;
                        break;
                    case pred::IUGE:
                        holds = x >= y;
                        break;
                    case pred::IULT:
                        holds = x < y;
                        break;
                    case pred::IULE:
                        holds = x <= y;
                        break;
                    case pred::SGT:
                        holds = sx > sy;
                        break;
                    case pred::SGE:
                        holds = sx >= sy;
                        break;
                    case pred::SLT:
                        holds = sx < sy;
                        break;
                    default:
                        holds = sx <= sy;
                        break;
                    }
                } else {
                    float fx = fl(x), fy = fl(y);
                    if (std::isnan(fx) || std::isnan(fy))
                        holds = (p & 8) != 0;
                    else
                        holds = ((p & 1) != 0 && fx == fy) || ((p & 2) != 0 && fx > fy) || ((p & 4) != 0 && fx < fy);
                }
                r = holds ? 1 : 0;
                break;
            }
            case Op::Select:
                r = eval(op.a) != 0 ? eval(op.b) : eval(op.c);
                break;
            case Op::Cast: {
                const auto from_bits = int_bits(m, m.type_of(op.a));
                std::uint64_t x = eval(op.a);
                const Cast co = Cast(op.code);
                if (co == Cast::Bitcast || co == Cast::ZExt)
                    r = x;
                else if (co == Cast::SExt && from_bits && bits)
                    r = std::uint64_t(signed_(*from_bits, x)) & mask(*bits);
                else if (co == Cast::Trunc && bits)
                    r = x & mask(*bits);
                else
                    bug("a cast the generator does not make");
                break;
            }
            case Op::Call: {
                std::vector<std::uint64_t> a;
                for (V v : op.list) a.push_back(eval(v));
                const std::string& name = m.funcs.at(op.index).name;
                if (name == "dx.op.loadInput.i32") {
                    r = inputs[std::size_t(a.at(0) ^ a.at(1)) % inputs.size()];
                } else if (name == "dx.op.binary.i32") {
                    r = std::min(a.at(0), a.at(1));
                } else if (name == "dx.op.cbufferLoadLegacy.i32") {
                    r = std::rotl(inputs[std::size_t(a.at(0)) % inputs.size()], int(std::uint32_t(a.at(1)) & 31));
                } else if (name == "dx.op.bufferLoad.i32") {
                    r = memory[std::size_t(a.at(0) ^ a.at(1)) % 16];
                } else if (name == "dx.op.waveReadLaneFirst.i32") {
                    r = a.at(0) ^ a.at(1);
                } else {
                    memory[std::size_t(a.at(0)) % 16] = a.at(1);
                    trace.emplace_back(name, a);
                    r = 0;
                }
                break;
            }
            default:
                bug("an instruction the generator does not make");
            }
            vals[i] = r;
        }
        from = b;
        if (!blk.term) bug("a terminator");
        const Term& t = *blk.term;
        switch (t.kind) {
        case Term::Ret:
            trace.emplace_back("ret", std::vector<std::uint64_t>{});
            return trace;
        case Term::Br:
            b = t.t;
            break;
        case Term::CondBr:
            b = eval(t.cond) != 0 ? t.t : t.f;
            break;
        }
    }
    trace.emplace_back("still running", std::vector<std::uint64_t>{});
    return trace;
}

// A random structured program over a few registers, as the lowering makes them: every register
// joined at every join, conditions tested again behind their branch, values computed again.
struct Gen {
    Kit k;
    std::uint64_t rng = 0;
    std::vector<V> regs;
    std::vector<V> conds;
    std::size_t cur = 0;
    V input, table, wave;
    // The operands of the block's last read (to read the place again, and to store to it).
    std::optional<std::pair<V, V>> last;

    std::uint64_t next(std::uint64_t n) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return (rng >> 11) % n;
    }

    V reg() {
        std::size_t n = std::size_t(next(regs.size()));
        return regs[n];
    }

    // A register or a small constant.
    V value() {
        if (next(4) == 0) {
            static constexpr std::uint32_t ks[] = {0u, 1, 2, 31, 0xFFFFFFFFu, 0x80000000u, 7};
            std::uint32_t kv = ks[next(7)];
            return k.m.i32(kv);
        }
        return reg();
    }

    V cond() {
        const std::size_t c = cur;
        V made;
        const std::uint64_t pick = next(6);
        if (pick == 0 && !conds.empty()) {
            std::size_t n = std::size_t(next(conds.size()));
            return conds[n];
        } else if (pick == 1 && conds.size() > 1) {
            std::uint64_t a = next(conds.size());
            std::uint64_t b = next(conds.size());
            static constexpr Bin ops[] = {Bin::And, Bin::Or, Bin::Xor};
            Bin op = ops[next(3)];
            made = k.m.bin(c, op, conds[a], conds[b]);
        } else if (pick == 2 && !conds.empty()) {
            std::size_t n = std::size_t(next(conds.size()));
            V t = k.m.bool_(true);
            made = k.m.bin(c, Bin::Xor, conds[n], t);
        } else {
            std::uint8_t p = std::uint8_t(pred::EQ + std::uint8_t(next(10)));
            V x = reg();
            V y = value();
            made = k.m.cmp(c, p, x, y);
        }
        conds.push_back(made);
        return made;
    }

    void statement() {
        const std::size_t c = cur;
        const std::size_t d = std::size_t(next(regs.size()));
        Ty i32t = k.m.int_(32);
        Ty f32t = k.m.float_();
        V r;
        const std::uint64_t pick = next(14);
        if (pick <= 3) {
            static constexpr Bin ops[] = {Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor};
            Bin op = ops[next(6)];
            V x = value();
            V y = value();
            r = k.m.bin(c, op, x, y);
        } else if (pick == 4) {
            static constexpr Bin ops[] = {Bin::Shl, Bin::LShr, Bin::AShr};
            Bin op = ops[next(3)];
            V x = value();
            std::uint32_t by = std::uint32_t(next(32));
            V byv = k.m.i32(by);
            r = k.m.bin(c, op, x, byv);
        } else if (pick == 5 || pick == 6) {
            // A guarded write: the old value where the guard fails.
            V g = cond();
            V x = value();
            r = k.m.select(cur, g, x, regs[d]);
        } else if (pick == 7) {
            V x = value();
            V y = value();
            const V fs[] = {k.pure, input, table, wave};
            V f = fs[next(4)];
            r = k.m.call(c, f, {x, y});
        } else if (pick == 8 || pick == 9) {
            const bool again = next(2) == 0;
            V x, y;
            if (last && again) {
                x = last->first;
                y = last->second;
            } else {
                x = value();
                y = value();
            }
            last = std::make_pair(x, y);
            r = k.m.call(c, k.reads, {x, y});
        } else if (pick == 10) {
            // A store, half the time to the place read last.
            V x = value();
            V y = value();
            const bool there = next(2) == 0;
            if (last && there) x = k.m.bin(c, Bin::Xor, last->first, last->second);
            k.m.call(c, k.writes, {x, y});
            return;
        } else if (pick == 11) {
            // Float arithmetic, read back as bits.
            V x = value();
            V y = value();
            V fx = k.m.cast(c, Cast::Bitcast, x, f32t);
            V fy = k.m.cast(c, Cast::Bitcast, y, f32t);
            static constexpr Bin ops[] = {Bin::Add, Bin::Sub, Bin::Mul};
            Bin op = ops[next(3)];
            V f = k.m.bin(c, op, fx, fy);
            if (next(2) == 0) {
                std::uint8_t p = std::uint8_t(next(16));
                V t = k.m.cmp(c, p, f, fy);
                conds.push_back(t);
            }
            r = k.m.cast(c, Cast::Bitcast, f, i32t);
        } else if (pick == 12) {
            V g = cond();
            r = k.m.cast(cur, Cast::ZExt, g, i32t);
        } else {
            V x = value();
            V f = k.m.cast(c, Cast::Bitcast, x, f32t);
            r = k.m.cast(c, Cast::Bitcast, f, i32t);
        }
        regs[d] = r;
    }

    void code(std::uint32_t depth) {
        const std::uint64_t n = 1 + next(6);
        for (std::uint64_t s = 0; s < n; ++s) {
            const std::uint64_t pick = next(8);
            if ((pick == 0 || pick == 1) && depth > 0)
                branch(depth - 1);
            else if (pick == 2 && depth > 0)
                repeat(depth - 1);
            else
                statement();
        }
    }

    // if / else, every register joined.
    void branch(std::uint32_t depth) {
        V c = cond();
        Ty i32t = k.m.int_(32);
        std::size_t then = k.m.new_block();
        std::size_t other = k.m.new_block();
        std::size_t join = k.m.new_block();
        k.m.terminate(cur, Term::cond_br(c, then, other));
        std::vector<V> before = regs;
        const std::size_t known = conds.size();
        cur = then;
        last.reset();
        code(depth);
        std::vector<V> then_regs = std::exchange(regs, before);
        const std::size_t then_end = cur;
        k.m.terminate(then_end, Term::br(join));
        conds.resize(std::min(conds.size(), known));
        cur = other;
        last.reset();
        if (next(3) != 0) code(depth);
        const std::size_t other_end = cur;
        k.m.terminate(other_end, Term::br(join));
        conds.resize(std::min(conds.size(), known));
        for (std::size_t r = 0; r < then_regs.size(); ++r) {
            V p = k.m.phi(join, i32t);
            k.m.set_phi(p, {{then_regs[r], then_end}, {regs[r], other_end}});
            regs[r] = p;
        }
        cur = join;
        last.reset();
    }

    // A loop that runs at most a few times, every register joined at its head.
    void repeat(std::uint32_t depth) {
        Ty i32t = k.m.int_(32);
        std::size_t head = k.m.new_block();
        std::size_t exit = k.m.new_block();
        const std::size_t before = cur;
        k.m.terminate(before, Term::br(head));
        std::vector<V> entering = regs;
        std::vector<V> joins;
        for (std::size_t r = 0; r < regs.size(); ++r) joins.push_back(k.m.phi(head, i32t));
        V count = k.m.phi(head, i32t);
        regs = joins;
        const std::size_t known = conds.size();
        cur = head;
        last.reset();
        code(depth);
        V one = k.m.i32(1);
        V nx = k.m.bin(cur, Bin::Add, count, one);
        std::uint32_t limit = 1 + std::uint32_t(next(3));
        V limitv = k.m.i32(limit);
        V more = k.m.cmp(cur, pred::IULT, nx, limitv);
        if (next(2) == 0) {
            V c = cond();
            more = k.m.bin(cur, Bin::And, more, c);
        }
        const std::size_t latch = cur;
        k.m.terminate(latch, Term::cond_br(more, head, exit));
        for (std::size_t r = 0; r < joins.size(); ++r) k.m.set_phi(joins[r], {{entering[r], before}, {regs[r], latch}});
        V zero = k.m.i32(0);
        k.m.set_phi(count, {{zero, before}, {nx, latch}});
        conds.resize(std::min(conds.size(), known));
        cur = exit;
        last.reset();
    }
};

Module random_program(std::uint64_t seed) {
    Gen g;
    g.k = kit();
    Ty i32t = g.k.m.int_(32);
    Ty fty = g.k.m.func_ty(i32t, {i32t, i32t});
    g.input = g.k.m.function("dx.op.loadInput.i32", fty, Attr::ReadNone, false);
    g.table = g.k.m.function("dx.op.cbufferLoadLegacy.i32", fty, Attr::ReadOnly, false);
    g.wave = g.k.m.function("dx.op.waveReadLaneFirst.i32", fty, Attr::NoUnwind, false);
    const std::size_t b = g.k.b;
    g.rng = (seed * 0x9E3779B97F4A7C15ull) | 1;
    g.cur = b;
    for (std::uint32_t r = 0; r < 5; ++r) {
        V x = g.k.m.i32(r);
        V y = g.k.m.i32(r * 3);
        V v = g.k.m.call(b, g.input, {x, y});
        g.regs.push_back(v);
    }
    g.code(3);
    g.code(2);
    // Every register is read at the end.
    for (std::size_t r = 0; r < g.regs.size(); ++r) {
        V at = g.k.m.i32(std::uint32_t(r));
        g.k.m.call(g.cur, g.k.writes, {at, g.regs[r]});
    }
    g.k.m.terminate(g.cur, Term::ret());
    return std::move(g.k.m);
}

void every_pass_leaves_what_random_programs_store() {
    std::size_t removed = 0;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (std::uint64_t seed = 1; seed <= 400; ++seed) {
        Module plain = random_program(seed);
        std::vector<std::vector<std::uint32_t>> inputs;
        for (std::uint32_t n = 0; n < 6; ++n) {
            std::vector<std::uint32_t> in;
            for (std::uint32_t i = 0; i < 7; ++i) {
                switch (n) {
                case 0:
                    in.push_back(0);
                    break;
                case 1:
                    in.push_back(i);
                    break;
                case 2:
                    in.push_back(0xFFFFFFFFu - i);
                    break;
                default:
                    in.push_back(std::rotl((std::uint32_t(seed) ^ n) * 0x9E3779B9u, int(i * 5)) ^ (i << 28));
                    break;
                }
            }
            inputs.push_back(std::move(in));
        }
        std::vector<Trace> expected;
        for (const auto& in : inputs) expected.push_back(run(plain, in));
        std::size_t written = 0;
        for (const Block& blk : plain.blocks) written += blk.body.size() + blk.phis.size();
        for (std::uint32_t passes : {DEAD, FOLD, REUSE, FACTS, LOADS, FOLD | FACTS, FOLD | REUSE | DEAD, ALL}) {
            Module m = random_program(seed);
            m.optimise(passes);
            for (std::size_t n = 0; n < inputs.size(); ++n) {
                bool same = run(m, inputs[n]) == expected[n];
                if (!same) std::fprintf(stderr, "seed %llu, passes %u\n", (unsigned long long)seed, passes);
                CHECK(same);
            }
            if (passes == ALL) {
                std::size_t left = 0;
                for (const Block& blk : m.blocks) left += blk.body.size() + blk.phis.size();
                removed += written - left;
            }
            std::vector<std::uint8_t> bytes = m.write();
            hash = (hash ^ test_util::fnv(bytes)) * 0x00000100000001b3ull;
        }
    }
    // The programs are of the wasteful kind: the passes find work.
    std::printf("random programs: %zu instructions removed, written hash %016llx\n", removed, (unsigned long long)hash);
    CHECK(removed > 4000);
    // The same as Rust's: the generator and the passes made the same modules, byte for byte.
    CHECK(removed == 44159);
    CHECK(hash == 0x7efd5f13281dc929ull);
}

int run_test(const char* name, void (*f)()) {
    int before = test_util::failures();
    current_test = name;
    try {
        f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: threw %s\n", name, e.what());
        ++test_util::failures();
    }
    if (test_util::failures() != before) std::fprintf(stderr, "%s: FAILED\n", name);
    return 0;
}

}  // namespace

int main() {
#define T(f) run_test(#f, f)
    T(unread_results_without_effect_are_dropped);
    T(branch_conditions_and_phi_cycles);
    T(wave_operations_have_no_effect_of_their_own);
    T(constants_and_identities_fold);
    T(float_arithmetic_is_not_folded);
    T(a_value_is_reused_below_its_block_and_not_beside_it);
    T(a_branch_taken_says_what_its_condition_is);
    T(a_branch_on_a_constant_goes_one_way);
    T(a_read_is_reused_within_its_block_until_a_store);
    T(a_join_of_one_value_and_two_equal_joins);
    T(mixed_cross_check);
    T(every_pass_leaves_what_random_programs_store);
#undef T
    CHECK(written_count == std::size(RUST_WRITTEN));
    return test_util::finish("opt_test");
}
