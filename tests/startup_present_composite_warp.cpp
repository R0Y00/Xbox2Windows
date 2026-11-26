// Pixel readback of the presented-plane composite on WARP (docs/startup-runtime.md#presented-planes).
// Source-built planes go through the code the output window uses: synthetic 0x48-byte plane records
// read by present_plane_resource, the order from composite_steps (present_planes.h), and the draws of
// PlaneCompositor (plane_compositor.h) into a 64 x 32 RGBA8 target that is read back.
//
// Contract checked: the bottom plane is drawn opaque, its alpha ignored; each later plane is blended
// over the result in plane order, colour = src * a + dst * (1 - a); the target alpha stays 255;
// repeats of a surface and planes whose record has no readable resource are left out.
//
// Cases:
//   baseline  one plane of four colours, one cell with alpha 0, shown as is.
//   stacked   black opaque plane 0; plane 1 with a fully transparent cell, a half-transparent cell,
//             a nearly transparent cell (alpha 5) and an opaque cell; plane 2 with an opaque cell, a
//             cell blended over plane 1's opaque cell (order) and transparent elsewhere.
//   skipped   plane 0 black, plane 1 as above, plane 2 the same surface as plane 1, plane 3 a record
//             with no resource: the same picture as plane 0 and 1 alone.
// --control runs the same cases and then removes the overlay: stacked composited from the black
// plane 0 alone. It prints "control ok:" and exits 0 only when the baseline and stacked cases matched,
// every fence completed without device removal, and the overlay-free result is black and differs from
// the stacked expectation in exactly the six cells the overlays draw. A setup failure, a missing
// fence, device removal or any other difference prints FAIL and exits 1.
// usage: startup_present_composite_warp [--control]   (exit 77 when no WARP adapter is available)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "startup_stubs/plane_compositor.h"
#include "startup_stubs/present_planes.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void fail(const char* format, const char* a, unsigned b, unsigned c, const unsigned char* got, const unsigned char* want)
{
    ++failures;
    std::printf(format, a, b, c, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
}

constexpr UINT width = 64, height = 32;
constexpr UINT columns = 4, rows = 2;  // cells of 16 x 16 texels
constexpr UINT cell_w = width / columns, cell_h = height / rows;

struct Rgba {
    unsigned char r, g, b, a;
};
// A plane picture: one colour per cell, row-major.
using Cells = Rgba[rows][columns];

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 value = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // Executes the list and waits for its fence. A list that does not close or execute, a fence that
    // does not reach the value in time, or one that reports device removal (UINT64_MAX) is a failure.
    bool run()
    {
        const HRESULT closed = list->Close();
        if (FAILED(closed)) {
            ++failures;
            std::printf("FAIL: command list close 0x%08lx\n", static_cast<unsigned long>(closed));
            return false;
        }
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        const HRESULT signalled = queue->Signal(fence, ++value);
        if (SUCCEEDED(signalled) && fence->GetCompletedValue() < value && SUCCEEDED(fence->SetEventOnCompletion(value, event)))
            WaitForSingleObject(event, 10000);
        const UINT64 completed = fence->GetCompletedValue();
        if (FAILED(signalled) || completed == UINT64_MAX || completed < value) {
            ++failures;
            std::printf("FAIL: fence %llu not completed (completed %llu, signal 0x%08lx)\n", static_cast<unsigned long long>(value),
                        static_cast<unsigned long long>(completed), static_cast<unsigned long>(signalled));
            return false;
        }
        ++fences;
        allocator->Reset();
        list->Reset(allocator, nullptr);
        return true;
    }
    unsigned fences = 0;  // fences that completed
};

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    list->ResourceBarrier(1, &b);
}

D3D12_RESOURCE_DESC texture_desc(D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

ID3D12Resource* buffer(Gpu& gpu, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* out = nullptr;
    gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out));
    return out;
}

// A plane surface holding `cells`, left in COMMON as the compositor expects.
ID3D12Resource* plane(Gpu& gpu, const Cells& cells)
{
    const D3D12_RESOURCE_DESC desc = texture_desc(D3D12_RESOURCE_FLAG_NONE);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* texture = nullptr;
    if (FAILED(gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&texture))))
        return nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    ID3D12Resource* upload = buffer(gpu, D3D12_HEAP_TYPE_UPLOAD, total, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    upload->Map(0, nullptr, &mapped);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            const Rgba& c = cells[y / cell_h][x / cell_w];
            unsigned char* t = static_cast<unsigned char*>(mapped) + footprint.Offset + std::size_t{y} * footprint.Footprint.RowPitch + x * 4;
            t[0] = c.r, t[1] = c.g, t[2] = c.b, t[3] = c.a;
        }
    upload->Unmap(0, nullptr);
    barrier(gpu.list, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION to{}, from{};
    to.pResource = texture;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.pResource = upload;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    gpu.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    barrier(gpu.list, texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    const bool ran = gpu.run();
    upload->Release();
    if (!ran) {
        texture->Release();
        return nullptr;
    }
    return texture;
}

// Plane records as a console present passes them: 0x48 bytes each, +0x08 the resource count and
// +0x10 a pointer to the resource array. A null surface gives a record with no resource.
struct Records {
    alignas(8) unsigned char bytes[4 * x2w_startup::present_plane_stride] = {};
    std::uint64_t arrays[4][1] = {};
    explicit Records(ID3D12Resource* const* surfaces, unsigned count)
    {
        for (unsigned i = 0; i < count; ++i) {
            unsigned char* record = bytes + i * x2w_startup::present_plane_stride;
            const std::uint64_t token = 7, resources = surfaces[i] ? 1 : 0;
            arrays[i][0] = reinterpret_cast<std::uint64_t>(surfaces[i]);
            const std::uint64_t pointer = reinterpret_cast<std::uint64_t>(arrays[i]);
            std::memcpy(record + 0x00, &token, 8);
            std::memcpy(record + 0x08, &resources, 8);
            std::memcpy(record + 0x10, &pointer, 8);
        }
    }
};

struct Target {
    ID3D12Resource* texture = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    ID3D12Resource* readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
};

// Runs the window's composite path on `count` plane records and reads the target back.
// Returns width * height RGBA texels, or nothing when the composite drew no plane.
std::vector<unsigned char> composite(Gpu& gpu, x2w_startup::PlaneCompositor& compositor, Target& target, ID3D12Resource* const* surfaces,
                                     unsigned count, unsigned* steps_out)
{
    Records records(surfaces, count);
    const auto readable = [&](const void* address, std::size_t bytes) {
        const auto* a = static_cast<const unsigned char*>(address);
        const auto* lo = reinterpret_cast<const unsigned char*>(&records);
        return a >= lo && a + bytes <= lo + sizeof(records);
    };
    // As queue_present: plane i's first resource, 0 when the record has none.
    std::uint64_t resources[4] = {};
    ID3D12Resource* plane_surfaces[4] = {};
    for (unsigned i = 0; i < count; ++i) {
        resources[i] = x2w_startup::present_plane_resource(records.bytes, count, i, readable);
        plane_surfaces[i] = reinterpret_cast<ID3D12Resource*>(resources[i]);
    }
    x2w_startup::PlaneStep steps[4];
    const unsigned n = x2w_startup::composite_steps(resources, count, steps, 4);
    *steps_out = n;
    barrier(gpu.list, target.texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const float clear[4] = {0.25f, 0.5f, 0.75f, 0.0f};  // nothing of this may survive the opaque base
    gpu.list->ClearRenderTargetView(target.heap->GetCPUDescriptorHandleForHeapStart(), clear, 0, nullptr);
    const UINT drawn = compositor.record(gpu.device, gpu.list, target.heap->GetCPUDescriptorHandleForHeapStart(), width, height, steps, n,
                                         plane_surfaces);
    barrier(gpu.list, target.texture, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION to{}, from{};
    to.pResource = target.readback;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = target.footprint;
    from.pResource = target.texture;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    std::vector<unsigned char> pixels;
    if (!gpu.run() || !drawn) return pixels;
    pixels.resize(std::size_t{width} * height * 4);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(target.footprint.Footprint.RowPitch) * height};
    target.readback->Map(0, &range, &mapped);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(&pixels[std::size_t{y} * width * 4],
                    static_cast<const unsigned char*>(mapped) + target.footprint.Offset + std::size_t{y} * target.footprint.Footprint.RowPitch,
                    std::size_t{width} * 4);
    const D3D12_RANGE none{0, 0};
    target.readback->Unmap(0, &none);
    return pixels;
}

// Whether every texel of a cell is within `tolerance` of `want` (alpha 255); the first texel off it
// goes to `got`.
bool cell_matches(const std::vector<unsigned char>& pixels, UINT row, UINT column, const Rgba& want, int tolerance, unsigned char* got)
{
    const unsigned char w[4] = {want.r, want.g, want.b, 255};
    for (UINT y = row * cell_h; y < (row + 1) * cell_h; ++y)
        for (UINT x = column * cell_w; x < (column + 1) * cell_w; ++x) {
            const unsigned char* g = &pixels[(std::size_t{y} * width + x) * 4];
            for (int k = 0; k < 4; ++k)
                if (std::abs(int{g[k]} - int{w[k]}) > tolerance) {
                    std::memcpy(got, g, 4);
                    return false;
                }
        }
    return true;
}

// Every texel of every cell must be within `tolerance` of the expected colour (alpha 255).
void expect(const char* name, const std::vector<unsigned char>& pixels, const Rgba (&want)[rows][columns], const int (&tolerance)[rows][columns])
{
    if (pixels.empty()) {
        ++failures;
        std::printf("FAIL: %s: nothing composited\n", name);
        return;
    }
    for (UINT row = 0; row < rows; ++row)
        for (UINT column = 0; column < columns; ++column) {
            const unsigned char w[4] = {want[row][column].r, want[row][column].g, want[row][column].b, 255};
            bool reported = false;
            for (UINT y = row * cell_h; y < (row + 1) * cell_h && !reported; ++y)
                for (UINT x = column * cell_w; x < (column + 1) * cell_w && !reported; ++x) {
                    const unsigned char* g = &pixels[(std::size_t{y} * width + x) * 4];
                    for (int k = 0; k < 4 && !reported; ++k)
                        if (std::abs(int{g[k]} - int{w[k]}) > tolerance[row][column]) {
                            fail("FAIL: %s cell row %u column %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", name, row, column, g, w);
                            reported = true;
                        }
                }
            if (!reported)
                std::printf("%s cell row %u column %u: %u,%u,%u,255 (+/-%d)\n", name, row, column, w[0], w[1], w[2], tolerance[row][column]);
        }
}

}  // namespace

int main(int argc, char** argv)
{
    const bool control = argc > 1 && std::strcmp(argv[1], "--control") == 0;
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter* warp = nullptr;
    Gpu gpu;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&gpu.device)))) {
        std::printf("SKIP: no WARP device\n");
        return 77;
    }
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(gpu.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&gpu.queue))) ||
        FAILED(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&gpu.allocator))) ||
        FAILED(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator, nullptr, IID_PPV_ARGS(&gpu.list))) ||
        FAILED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence)))) {
        std::printf("FAIL: WARP queue setup\n");
        return 1;
    }
    x2w_startup::PlaneCompositor compositor;
    const HRESULT created = compositor.create(gpu.device, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
    if (FAILED(created)) {
        std::printf("FAIL: compositor setup 0x%08lx\n", static_cast<unsigned long>(created));
        return 1;
    }

    Target target;
    {
        const D3D12_RESOURCE_DESC desc = texture_desc(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
                                            IID_PPV_ARGS(&target.texture));
        const D3D12_DESCRIPTOR_HEAP_DESC rtv{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        gpu.device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&target.heap));
        gpu.device->CreateRenderTargetView(target.texture, nullptr, target.heap->GetCPUDescriptorHandleForHeapStart());
        UINT64 total = 0;
        gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &target.footprint, nullptr, nullptr, &total);
        target.readback = buffer(gpu, D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    if (!target.texture || !target.heap || !target.readback) {
        std::printf("FAIL: target setup\n");
        return 1;
    }

    // Plane pictures. The baseline plane's last cell has alpha 0 to show that the bottom plane's
    // alpha is ignored.
    const Cells baseline = {{{200, 40, 10, 255}, {10, 200, 40, 255}, {40, 10, 200, 128}, {250, 250, 250, 255}},
                            {{0, 0, 0, 255}, {128, 64, 32, 255}, {1, 2, 3, 255}, {90, 180, 60, 0}}};
    const Cells black = {{{0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}},
                         {{0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}}};
    // Plane 1: row 0 transparent (alpha 0), half (128), nearly transparent (5), opaque; row 1 opaque
    // blue under plane 2's blended cell, opaque blue under plane 2's opaque cell, transparent twice.
    const Cells overlay = {{{255, 0, 0, 0}, {0, 255, 0, 128}, {255, 255, 255, 5}, {0, 0, 255, 255}},
                           {{0, 0, 255, 255}, {0, 0, 255, 255}, {255, 255, 255, 0}, {255, 255, 255, 0}}};
    // Plane 2: row 0 transparent; row 1 half red over blue (order), opaque yellow over blue, opaque
    // cyan over black, transparent.
    const Cells top = {{{255, 255, 0, 0}, {255, 255, 0, 0}, {255, 255, 0, 0}, {255, 255, 0, 0}},
                       {{255, 0, 0, 128}, {255, 255, 0, 255}, {0, 255, 255, 255}, {255, 0, 255, 0}}};
    ID3D12Resource* const p_baseline = plane(gpu, baseline);
    ID3D12Resource* const p_black = plane(gpu, black);
    ID3D12Resource* const p_overlay = plane(gpu, overlay);
    ID3D12Resource* const p_top = plane(gpu, top);
    if (!p_baseline || !p_black || !p_overlay || !p_top) {
        std::printf("FAIL: plane setup\n");
        return 1;
    }

    const int exact[rows][columns] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    unsigned steps = 0;

    // Baseline: one plane, shown as is, alpha forced to 255.
    {
        ID3D12Resource* const planes[] = {p_baseline};
        const auto pixels = composite(gpu, compositor, target, planes, 1, &steps);
        Rgba want[rows][columns];
        for (UINT r = 0; r < rows; ++r)
            for (UINT c = 0; c < columns; ++c) want[r][c] = baseline[r][c];
        if (steps != 1) {
            ++failures;
            std::printf("FAIL: baseline: %u steps, want 1\n", steps);
        }
        expect("baseline", pixels, want, exact);
    }

    const HRESULT composite_created = compositor.create(gpu.device, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    if (FAILED(composite_created)) {
        std::printf("FAIL: multi-plane compositor setup 0x%08lx\n", static_cast<unsigned long>(composite_created));
        return 1;
    }

    // Stacked: black, overlay, top.
    // Expected (UNORM8, src * a/255 + dst * (1 - a/255)): 128/255 of 255 is 128; 5/255 of 255 is 5;
    // half red over blue is 128,0,127.
    const Rgba stacked_want[rows][columns] = {{{0, 0, 0, 255}, {0, 128, 0, 255}, {5, 5, 5, 255}, {0, 0, 255, 255}},
                                              {{128, 0, 127, 255}, {255, 255, 0, 255}, {0, 255, 255, 255}, {0, 0, 0, 255}}};
    const int blended[rows][columns] = {{0, 1, 1, 0}, {1, 0, 0, 0}};
    {
        ID3D12Resource* const planes[] = {p_black, p_overlay, p_top};
        const auto pixels = composite(gpu, compositor, target, planes, 3, &steps);
        if (steps != 3) {
            ++failures;
            std::printf("FAIL: stacked: %u steps, want 3\n", steps);
        }
        expect("stacked", pixels, stacked_want, blended);
    }

    // Skipped: plane 2 repeats plane 1's surface and plane 3 has no resource. Drawing the repeat
    // would blend the half-green cell twice (0,191,0) and the nearly transparent one twice (10).
    {
        ID3D12Resource* const planes[] = {p_black, p_overlay, p_overlay, nullptr};
        const auto pixels = composite(gpu, compositor, target, planes, 4, &steps);
        if (steps != 2) {
            ++failures;
            std::printf("FAIL: skipped: %u steps, want 2 (plane 0 and plane 1)\n", steps);
        }
        const Rgba want[rows][columns] = {{{0, 0, 0, 255}, {0, 128, 0, 255}, {5, 5, 5, 255}, {0, 0, 255, 255}},
                                          {{0, 0, 255, 255}, {0, 0, 255, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}}};
        const int tolerance[rows][columns] = {{0, 1, 1, 0}, {0, 0, 0, 0}};
        expect("skipped", pixels, want, tolerance);
    }

    // Control: the overlays removed, stacked is composited from the black plane 0 alone. That must be
    // black everywhere, and must differ from the stacked expectation in exactly the cells the overlays
    // draw: every cell except row 0 column 0 (plane 1 transparent) and row 1 column 3 (both transparent).
    unsigned control_differences = 0;
    if (control && !failures) {
        const bool overlay_drawn[rows][columns] = {{false, true, true, true}, {true, true, true, false}};
        const Rgba black_want = {0, 0, 0, 255};
        ID3D12Resource* const planes[] = {p_black};
        const auto pixels = composite(gpu, compositor, target, planes, 1, &steps);
        if (steps != 1) {
            ++failures;
            std::printf("FAIL: control: overlay removed: %u steps, want 1\n", steps);
        }
        if (pixels.empty()) {
            ++failures;
            std::printf("FAIL: control: overlay removed: nothing composited\n");
        }
        for (UINT row = 0; row < rows && !pixels.empty(); ++row)
            for (UINT column = 0; column < columns; ++column) {
                unsigned char got[4] = {};
                const unsigned char black[4] = {0, 0, 0, 255};
                if (!cell_matches(pixels, row, column, black_want, 0, got)) {
                    fail("FAIL: %s cell row %u column %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", "control: overlay removed", row, column, got,
                         black);
                    continue;
                }
                const bool differs = !cell_matches(pixels, row, column, stacked_want[row][column], blended[row][column], got);
                if (differs) ++control_differences;
                if (differs != overlay_drawn[row][column]) {
                    ++failures;
                    std::printf("FAIL: control: overlay removed: cell row %u column %u %s the stacked expectation, want it to %s\n", row,
                                column, differs ? "differs from" : "matches", overlay_drawn[row][column] ? "differ" : "match");
                } else {
                    std::printf("control: overlay removed: cell row %u column %u %s the stacked expectation\n", row, column,
                                differs ? "differs from" : "matches");
                }
            }
    }

    const HRESULT removed = gpu.device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        ++failures;
        std::printf("FAIL: device removed 0x%08lx\n", static_cast<unsigned long>(removed));
    }
    if (failures) {
        std::printf("startup_present_composite: FAIL (%d)\n", failures);
        return 1;
    }
    if (control) {
        std::printf("control ok: baseline matched, %u fences completed, removing the overlay changed %u of %u cells as expected\n",
                    gpu.fences, control_differences, rows * columns);
        return 0;
    }
    std::puts("startup_present_composite: ok (baseline, stacked and skipped planes on WARP)");
    return 0;
}
