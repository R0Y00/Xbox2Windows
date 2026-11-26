// Port of the tests in the upstream Rust module emit/bitcode.rs (#[cfg(test)] mod tests).

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "x2w_shader/emit/bitcode.h"
#include "test_util.h"

using namespace x2w::vs::bitcode;

namespace {

// A structured shader's shape: `stages` if/else diamonds in a row, each merge with one phi per
// register (as the lowering makes them), then a loop, and a tail that reads only the first `read`
// registers.
Module synthetic(std::size_t regs, std::size_t stages, std::size_t read) {
    Module m;
    Ty i32t = m.int_(32);
    Ty voidt = m.void_();
    Ty fty = m.func_ty(i32t, {i32t, i32t});
    V f = m.function("dx.op.synthetic", fty, Attr::ReadNone, false);
    Ty main_ty = m.func_ty(voidt, {});
    m.function("main", main_ty, std::nullopt, true);
    std::size_t b = m.new_block();
    std::vector<V> cur;
    for (std::size_t r = 0; r < regs; ++r) cur.push_back(m.i32(std::uint32_t(r)));
    for (std::size_t s = 0; s < stages; ++s) {
        V c = m.cmp(b, pred::EQ, cur[0], cur[s % regs]);
        std::size_t t = m.new_block();
        std::size_t e = m.new_block();
        std::size_t j = m.new_block();
        m.terminate(b, Term::cond_br(c, t, e));
        V x = m.call(t, f, {cur[s % regs], cur[(s + 1) % regs]});
        m.terminate(t, Term::br(j));
        m.terminate(e, Term::br(j));
        std::vector<V> next;
        for (std::size_t r = 0; r < regs; ++r) {
            V p = m.phi(j, i32t);
            V tv = r == s % regs ? x : cur[r];
            m.set_phi(p, {{tv, t}, {cur[r], e}});
            next.push_back(p);
        }
        cur = next;
        b = j;
    }
    // A loop: header phis fed back from an empty latch (which is folded).
    std::size_t head = m.new_block();
    std::size_t body = m.new_block();
    std::size_t latch = m.new_block();
    std::size_t exit = m.new_block();
    m.terminate(b, Term::br(head));
    std::vector<V> phis;
    for (std::size_t r = 0; r < regs; ++r) phis.push_back(m.phi(head, i32t));
    V one = m.i32(1);
    std::vector<V> stepped;
    for (V p : phis) stepped.push_back(m.bin(body, Bin::Add, p, one));
    V c = m.cmp(body, pred::IULT, stepped[0], cur[0]);
    m.terminate(head, Term::br(body));
    m.terminate(body, Term::cond_br(c, latch, exit));
    m.terminate(latch, Term::br(head));
    for (std::size_t r = 0; r < regs; ++r) m.set_phi(phis[r], {{cur[r], b}, {stepped[r], latch}});
    V acc = stepped[0];
    for (std::size_t k = 1; k < std::max<std::size_t>(read, 1); ++k) acc = m.bin(exit, Bin::Add, acc, stepped[k]);
    (void)m.call(exit, f, {acc, cur[0]});
    m.terminate(exit, Term::ret());
    return m;
}

// Liveness as write computed it before the worklist: whole passes over the instructions until
// one changes nothing.
std::vector<bool> live_by_passes(const Module& m) {
    std::vector<bool> live(m.insts.size(), false);
    for (const Block& b : m.blocks) {
        for (std::uint32_t i : b.body) live[i] = true;
        if (b.term && b.term->kind == Term::CondBr && b.term->cond.kind == V::Inst) live[b.term->cond.i] = true;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < m.insts.size(); ++i) {
            if (!live[i]) continue;
            for_each_operand(m.insts[i].op, [&](V v) {
                if (v.kind == V::Inst && !live[v.i]) {
                    live[v.i] = true;
                    changed = true;
                }
            });
        }
    }
    return live;
}

void vbr_and_blocks() {
    Writer w;
    w.vbr(5, 6);
    w.vbr(100, 6);
    CHECK(w.cur == (5 | (0b100100ull << 6) | (3ull << 12)));
    Writer w2;
    w2.enter(8, 3);
    w2.record(1, {1});
    w2.exit();
    auto bytes = w2.finish();
    CHECK(bytes.size() % 4 == 0);
    CHECK(test_util::u32le(bytes, 4) == 1);
    CHECK(signed_vbr(-3) == 7);
    CHECK(signed_vbr(2) == 4);
}

void worklist_liveness_matches_passes() {
    const std::size_t cases[][3] = {{1, 0, 1}, {1, 1, 1}, {4, 3, 1}, {8, 17, 3}, {16, 40, 16}};
    for (auto& c : cases) {
        Module m = synthetic(c[0], c[1], c[2]);
        CHECK(m.live() == live_by_passes(m));
    }
}

void unread_phis_are_dropped() {
    Module m;
    Ty i32t = m.int_(32);
    std::size_t entry = m.new_block();
    std::size_t join = m.new_block();
    V a = m.i32(1);
    V b = m.i32(2);
    m.terminate(entry, Term::br(join));
    V read = m.phi(join, i32t);
    V unread = m.phi(join, i32t);
    // A cycle of phis that only feed each other stays dead.
    V cyc_a = m.phi(join, i32t);
    V cyc_b = m.phi(join, i32t);
    m.set_phi(read, {{a, entry}});
    m.set_phi(unread, {{b, entry}});
    m.set_phi(cyc_a, {{cyc_b, entry}});
    m.set_phi(cyc_b, {{cyc_a, entry}});
    V sum = m.bin(join, Bin::Add, read, b);
    m.terminate(join, Term::ret());
    auto live = m.live();
    CHECK(live[read.i] && live[sum.i]);
    CHECK(!live[unread.i] && !live[cyc_a.i] && !live[cyc_b.i]);
    CHECK(live == live_by_passes(m));
}

std::vector<std::uint8_t> latch_loop(bool with_dead_phi) {
    Module m;
    Ty i32t = m.int_(32);
    Ty voidt = m.void_();
    Ty fty = m.func_ty(voidt, {i32t});
    V f = m.function("sink", fty, Attr::NoUnwind, false);
    Ty main_ty = m.func_ty(voidt, {});
    m.function("main", main_ty, std::nullopt, true);
    std::size_t entry = m.new_block();
    std::size_t head = m.new_block();
    std::size_t body = m.new_block();
    std::size_t latch = m.new_block();
    std::size_t exit = m.new_block();
    V z = m.i32(0);
    V one = m.i32(1);
    V ten = m.i32(10);
    m.terminate(entry, Term::br(head));
    V i = m.phi(head, i32t);
    m.terminate(head, Term::br(body));
    V next = m.bin(body, Bin::Add, i, one);
    V cond = m.cmp(body, pred::IULT, next, ten);
    m.terminate(body, Term::cond_br(cond, latch, exit));
    if (with_dead_phi) {
        V dead = m.phi(latch, i32t);
        m.set_phi(dead, {{next, body}});
    }
    m.terminate(latch, Term::br(head));
    m.set_phi(i, {{z, entry}, {next, latch}});
    (void)m.call(exit, f, {next});
    m.terminate(exit, Term::ret());
    return m.write();
}

void empty_latch_with_dead_phis_is_folded() {
    // Two identical loops except one has an unread phi in its empty latch; both must fold the
    // latch and emit identical bitcode.
    CHECK(latch_loop(false) == latch_loop(true));
}

void successors_of_each_terminator() {
    CHECK(successors(std::nullopt).empty());
    CHECK(successors(Term::ret()).empty());
    auto s = successors(Term::br(7));
    CHECK(s.size() == 1 && s.to[0] == 7);
    auto c = successors(Term::cond_br(V::constant(0), 3, 5));
    CHECK(c.size() == 2 && c.to[0] == 3 && c.to[1] == 5);
}

// The written bytes are pinned to what the reference writer produces (its written_bytes_unchanged).
void written_bytes_unchanged() {
    struct Case {
        std::size_t regs, stages, read, len;
        std::uint64_t hash;
    };
    const Case cases[] = {{64, 64, 64, 38916, 0x9586730cf9534a45ull}, {128, 256, 32, 295952, 0x89e5de2d64f08785ull}};
    for (const Case& c : cases) {
        auto bytes = synthetic(c.regs, c.stages, c.read).write();
        std::printf("synthetic %zux%zu: %zu bytes, fnv %016llx\n", c.regs, c.stages, bytes.size(),
                    (unsigned long long)test_util::fnv(bytes));
        CHECK(bytes.size() == c.len);
        CHECK(test_util::fnv(bytes) == c.hash);
    }
}

}  // namespace

int main() {
    vbr_and_blocks();
    worklist_liveness_matches_passes();
    unread_phis_are_dropped();
    empty_latch_with_dead_phis_is_folded();
    successors_of_each_terminator();
    written_bytes_unchanged();
    return test_util::finish("bitcode_test");
}
