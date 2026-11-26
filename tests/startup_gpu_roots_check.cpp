// X2W_GPU_ABI (WP3 vbind): the per-list root record (src/startup_stubs/gpu_roots.h) stores the root arguments defined in the upstream
// State (vg/draw.rs :28-52, :955-1002): any number of root parameters and constant dwords, graphics and compute
// apart, 32 vertex slots; and the console topologies map as draw.rs set_topology does.
//
// --mutate is the negative control: the same checks against X2W's old Pass record limits (16 parameters of 8 dwords,
// 8 vertex slots), which must fail.
#include "startup_stubs/gpu_roots.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

bool mutate = false;
using namespace x2w_vbind;

// The record under test; with --mutate, X2W's old limits applied on top (what Pass.constants[16][8] / vb[8] keep).
void set_constants(Roots& r, bool compute, std::uint32_t slot, const std::uint32_t* v, std::uint32_t n, std::uint32_t offset)
{
    if (mutate) {
        if (slot >= 16 || offset >= 8) return;
        if (n > 8 - offset) n = 8 - offset;
    }
    r.set_constants(compute, slot, v, n, offset);
}
void set_vertex(Roots& r, std::uint32_t start, std::uint32_t n, const VertexView* v)
{
    if (mutate) {
        if (start >= 8) return;
        if (n > 8 - start) n = 8 - start;
    }
    r.set_vertex(start, n, v);
}
}  // namespace

int main(int argc, char** argv)
{
    mutate = argc > 1 && std::strcmp(argv[1], "--mutate") == 0;
    Roots r;

    // More than 8 dwords, past parameter 16.
    std::uint32_t values[40];
    for (std::uint32_t i = 0; i < 40; ++i) values[i] = 0x1000 + i;
    set_constants(r, false, 20, values, 12, 3);
    {
        const GpuRoot* p = r.at(false, 20);
        CHECK(p && p->kind == GpuRootKind::constants);
        CHECK(p && p->dwords.size() == 15);
        CHECK(p && p->dwords[0] == 0 && p->dwords[3] == 0x1000 && p->dwords[14] == 0x100b);
    }
    // 40 dwords at parameter 2 (the 64-dword capacity must retain entries beyond the first eight).
    set_constants(r, false, 2, values, 40, 0);
    CHECK(r.at(false, 2) && r.at(false, 2)->dwords.size() == 40 && r.at(false, 2)->dwords[39] == 0x1027);
    // A later partial write keeps the rest (draw.rs: c[offset..end] = values).
    const std::uint32_t one = 7;
    set_constants(r, false, 2, &one, 1, 30);
    CHECK(r.at(false, 2) && r.at(false, 2)->dwords[30] == 7 && r.at(false, 2)->dwords[29] == 0x101d);

    // Graphics and compute apart.
    set_constants(r, true, 2, &one, 1, 0);
    CHECK(r.at(true, 2) && r.at(true, 2)->dwords.size() == 1 && r.at(true, 2)->dwords[0] == 7);
    CHECK(r.at(false, 2) && r.at(false, 2)->dwords.size() == 40);
    r.set_table(true, 5, 0xabc000);
    CHECK(!r.at(false, 5) || r.at(false, 5)->kind == GpuRootKind::none);
    CHECK(r.at(true, 5) && r.at(true, 5)->kind == GpuRootKind::table && r.at(true, 5)->handle == 0xabc000);

    // A parameter that held a table becomes constants afresh (no stale dwords), and a view replaces constants.
    r.set_table(false, 20, 0x5000);
    set_constants(r, false, 20, &one, 1, 1);
    CHECK(r.at(false, 20) && r.at(false, 20)->kind == GpuRootKind::constants && r.at(false, 20)->dwords.size() == 2 &&
          r.at(false, 20)->dwords[0] == 0);
    r.set_view(false, 20, 0x123456789aull, GpuRootKind::uav);
    CHECK(r.at(false, 20) && r.at(false, 20)->kind == GpuRootKind::uav && r.at(false, 20)->address == 0x123456789aull &&
          r.at(false, 20)->dwords.empty());

    // 32 vertex slots; a null array clears (draw.rs clear_vertex_buffers).
    VertexView vv[4] = {{0x1000, 64, 16}, {0x2000, 32, 8}, {0x3000, 16, 4}, {0x4000, 8, 4}};
    set_vertex(r, 28, 4, vv);
    CHECK(r.vertex[31].address == 0x4000 && r.vertex[28].stride == 16);
    set_vertex(r, 29, 2, nullptr);
    CHECK(!mutate ? r.vertex[29].address == 0 && r.vertex[30].address == 0 && r.vertex[31].address == 0x4000 : true);
    r.set_vertex(30, 10, vv);  // past 32: cut, no write past the array
    CHECK(r.vertex[31].address == 0x2000);

    // Topologies (draw.rs :923, :4356): rectangles and the other console-only ones draw as triangle lists.
    CHECK(host_topology(topology_rectangles) == 4);
    CHECK(host_topology(topology_quads) == 4 && host_topology(topology_quad_strip) == 4);
    CHECK(host_topology(topology_triangle_fan) == 4 && host_topology(topology_line_loop) == 4);
    CHECK(host_topology(5) == 5 && host_topology(1) == 1);

    // A reset forgets everything.
    r.clear();
    CHECK(r.graphics.empty() && r.compute.empty() && r.vertex[31].address == 0);

    {
        x2w_gpu::TableConstants pending;
        unsigned char title[] = {1, 2, 3, 4};
        unsigned char upload[8]{};
        auto* revoked = static_cast<unsigned char*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        CHECK(revoked);
        if (revoked) {
            const auto keep = [&](const void* source, UINT at) {
                upload[at] = *pending.remember(source, 1, {1, 3, at});
            };
            keep(title, 1);
            keep(title + 1, 2);
            keep(revoked, 3);
            keep(title + 3, 4);
            DWORD old = 0;
            CHECK(VirtualProtect(revoked, 4096, PAGE_NOACCESS, &old));
            title[0] = 7;
            title[3] = 9;
            const auto write = [&](const x2w_gpu::TableConstantSlice& slice, const unsigned char* src, size_t len) {
                if (slice.offset == 4) return false; // its ring generation has retired
                CHECK(slice.chunk == 1 && slice.generation == 3 && len == 1);
                upload[slice.offset] = *src;
                return true;
            };
            const auto first = pending.first_submit(write);
            CHECK(first.checked == 4 && first.changed == 1 && first.unreadable == 1 && first.stale == 1);
            CHECK(upload[0] == 0 && upload[1] == 7 && upload[2] == 2 && upload[3] == 0 && upload[4] == 4 && upload[5] == 0);
            title[0] = 9;
            CHECK(!pending.first_submit(write).checked && upload[1] == 7);
            keep(title, 1);
            pending.clear(); // reset without submission
            CHECK(!pending.first_submit(write).checked);
            VirtualFree(revoked, 0, MEM_RELEASE);
        }
    }
    std::printf("%s: %d failure(s)\n", mutate ? "gpu roots (mutated control)" : "gpu roots", failures);
    return failures ? 1 : 0;
}
