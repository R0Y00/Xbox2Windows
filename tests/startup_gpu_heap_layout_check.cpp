// startup_gpu_heap_layout_check: the mirror heap layout and descriptor rules
// (src/startup_stubs/gpu_heap.h), without a device. The vectors are the upstream Rust tests in
// src/descriptor/host.rs where it has them.
//
//   startup_gpu_heap_layout_check                 every check passes
//   startup_gpu_heap_layout_check --mutate probe  the host refuses 2^22 views: no twins, so the
//                                                     layout checks fail (negative control)
//   startup_gpu_heap_layout_check --mutate most   X2W_SHADERCFG_DYNAMIC_SLOTS = 2^18: the dynamic
//                                                     range of X2W's title heap is another one
#include "startup_stubs/gpu_heap.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace gpu_heap;

static int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

static bool same(const std::optional<TwinWindow>& w, u64 first, u32 dwords, u32 by)
{
    return w && w->first == first && w->dwords == dwords && w->by == by;
}

int main(int argc, char** argv)
{
    std::string mutate;
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--mutate")) mutate = argv[i + 1];

    // probe_view_limit: an AMD host takes 2^22 (the 9070 XT does); WARP takes 2^21 only.
    const u32 host_most = mutate == "probe" ? (1u << 21) : (1u << 22);
    const Limits amd = probe_view_limit(true, [&](u32 n) { return n <= host_most; });
    CHECK(amd.raw == (1u << 21));
    CHECK(amd.null() == (1u << 21) - 1);
    const Limits warp = probe_view_limit(true, [](u32 n) { return n <= (1u << 21); });
    CHECK(warp.raw == 0 && warp.null() == (1u << 21) - 1);
    const Limits nvidia = probe_view_limit(true, [](u32 n) { return n <= portable_views; });
    CHECK(nvidia.raw == 0 && nvidia.null() == portable_views - 1);
    const Limits off = probe_view_limit(false, [](u32) { return true; });
    CHECK(off.raw == 0 && off.null() == (1u << 21) - 1);

    // dynamic_slots.
    CHECK(dynamic_slots(4, portable_views - 1) == dynamic_base);
    CHECK(dynamic_slots(4, (1u << 21) - 1) == dynamic_most_default);
    CHECK(dynamic_slots(2'000'000, (1u << 21) - 1) == dynamic_base);  // little room: at least 2^18

    // A synthetic heap large enough to reduce dynamic capacity.
    const u32 most = mutate == "most" ? dynamic_base : dynamic_most_default;
    const u32 title = (1u << 20) + (1u << 18);
    const Layout l = layout_for(false, title, amd, most);
    CHECK(l.size == (1u << 22));
    CHECK(l.null == (1u << 21) - 1);
    CHECK(l.raw == (1u << 21));
    CHECK(l.count == title && !l.capped);
    CHECK(l.dynamic == (1u << 21) - 1 - title);  // remaining space below the null slot
    CHECK(l.count + l.dynamic <= l.null);
    CHECK(l.twins == l.count + l.dynamic);
    for (u32 i : {0u, 1u, 70u, title - 1, title, title + l.dynamic - 1}) {
        CHECK(l.has_twin(i));
        CHECK(l.twin(i) == (1u << 21) + i);
        CHECK(l.twin(i) < l.size);
    }
    CHECK(!l.has_twin(l.twins));
    CHECK(l.is_dynamic(title) && l.is_dynamic(title + l.dynamic - 1));
    CHECK(!l.is_dynamic(title - 1) && !l.is_dynamic(title + l.dynamic));
    CHECK(l.clamp(l.size + 17) == l.null);
    CHECK(l.clamp(l.null) == l.null && l.clamp(5) == 5);
    CHECK(l.null_views_end() == (1u << 21));

    // A title heap over 2^21: indices beyond the limit resolve to the null slot.
    const Layout big = layout_for(false, (1u << 21) + 1, amd, most);
    CHECK(big.capped);
    CHECK(big.count + big.dynamic == big.null);
    CHECK(big.size == (1u << 22));
    // WARP / without twins: the heap ends at the null slot.
    const Layout w = layout_for(false, 1000, warp, most);
    CHECK(w.size == (1u << 21) && w.raw == 0 && w.twins == 0 && !w.has_twin(0));
    // Sampler heaps: no null slot, no dynamic, no twins.
    const Layout s = layout_for(true, 2048, amd, most);
    CHECK(s.size == 2048 && s.dynamic == 0 && s.raw == 0 && s.null == 0);

    // Increments 32 / 16 / 128 / 128.
    CHECK(increment(0) == 32u && increment(1) == 16u && increment(2) == 128u && increment(3) == 128u);
    CHECK(!increment(4));

    // host.rs a_structured_view_made_raw_passes_the_host_check.
    const auto rw = raw_window(32, 8, 100, 20);
    CHECK(rw && rw->first == 48 && rw->dwords == 500);
    CHECK(buffer_view_ok(format_r32_typeless, 48, 500, 0, true, 4096));
    CHECK(!buffer_view_ok(format_unknown, 48, 500, 0, true, 4096));
    CHECK(!buffer_view_ok(format_r32_typeless, 48, 500, 0, true, 2000));
    CHECK(!raw_window(4, 0, 100, 20));
    CHECK(!buffer_view_ok(format_r32_typeless, 0, 0, 0, true, 4096));
    CHECK(buffer_view_ok(format_unknown, 1, 10, 12, false, 4096));
    CHECK(!buffer_view_ok(format_unknown, 1, 10, 2052, false, 1 << 20));  // stride > 2048
    CHECK(!buffer_view_ok(27, 0, 4, 0, false, 4096));                    // typeless R8G8B8A8
    CHECK(buffer_view_ok(28, 0, 4, 0, false, 16));
    CHECK(!buffer_view_ok(71, 0, 4, 0, false, 4096));                    // BC1: blocks

    // host.rs raw_twins_show_the_views_bytes.
    CHECK(same(twin_window(42, 4, 10, 0, false, 4096), 4, 10, 0));
    CHECK(same(twin_window(57, 8, 7, 0, false, 4096), 4, 4, 0));
    CHECK(same(twin_window(57, 8, 7, 0, false, 30), 4, 3, 0));
    CHECK(same(twin_window(0, 2, 3, 24, false, 4096), 12, 18, 0));
    CHECK(same(twin_window(39, 8, 5, 0, true, 4096), 8, 5, 0));
    CHECK(same(twin_window(0, 1, 3, 24, false, 4096), 4, 20, 8));
    CHECK(same(twin_window(42, 1, 10, 0, false, 4096), 0, 11, 4));
    CHECK(!twin_window(42, 1, 10, 0, false, 4));

    // host.rs constant_bytes.
    CHECK(constant_bytes(480) == 512);
    CHECK(constant_bytes(64u << 10) == (64u << 10));
    CHECK(constant_bytes(~0u) == (64u << 10));

    // host.rs off_elements vectors.
    CHECK(off_elements(8, 0, true, 4));
    CHECK(!off_elements(0, 0, true, 4));
    CHECK(!off_elements(48, 0, true, 4));
    CHECK(!off_elements(456, 456, false, 0));
    CHECK(!off_elements(32, 456, false, 0));
    CHECK(off_elements(100, 456, false, 0));
    CHECK(off_elements(23'702'896, 0, false, 12));
    CHECK(!off_elements(48, 0, false, 12));
    CHECK(!off_elements(12, 0, false, 4));
    CHECK(off_elements(12, 0, false, 8));

    // The table of strides: structured the stride, raw 1, typed the format's, integer kinds.
    CHECK(stride_word(std::nullopt) == 0);
    CHECK(stride_word(Elements{456, false, 0}) == 456);
    CHECK(stride_word(Elements{0, true, 39}) == 1);
    CHECK(stride_word(Elements{0, false, 2}) == 16);                       // R32G32B32A32_FLOAT
    CHECK(stride_word(Elements{0, false, 42}) == (4 | view_integer));      // R32_UINT
    CHECK(stride_word(Elements{0, false, 43}) == (4 | view_integer | view_signed));  // R32_SINT
    CHECK(stride_word(Elements{0, false, 57}) == (2 | view_integer));      // R16_UINT
    CHECK((stride_word(Elements{0, false, 62}) & view_stride_bits) == 1);  // R8_UINT
    CHECK(pc_format(118) == 45 && pc_format(28) == 28);
    CHECK(family(29) == 27 && family(87) == 90 && family(100) == 100);

    // host.rs changed_records_are_those_that_differ_one_by_one.
    for (std::size_t step : {16u, 32u}) {
        const std::size_t count = 300;
        u64 seed = 7;
        auto random = [&]() {
            seed = seed * 0x5851F42D4C957F2Dull + 0x14057B7EF767814Full;
            return static_cast<std::size_t>(seed >> 33);
        };
        std::vector<unsigned char> last(step * count);
        for (auto& b : last) b = static_cast<unsigned char>(random());
        std::vector<unsigned char> now = last;
        std::vector<u32> differ;
        for (std::size_t slot = 0; slot < count; ++slot)
            if (random() % 9 == 0) {
                const std::size_t at = slot * step + random() % step;
                now[at] ^= static_cast<unsigned char>(1 + random() % 255);
                differ.push_back(1000 + static_cast<u32>(slot));
            }
        std::vector<u32> out;
        changed_records(last.data(), now.data(), now.size(), step, 1000, out);
        CHECK(out == differ);
        CHECK(last == now);
        out.clear();
        changed_records(last.data(), now.data(), now.size(), step, 1000, out);
        CHECK(out.empty());
    }

    // host.rs dynamic_slots_are_held_until_released, on a heap of 4 title slots.
    {
        DynamicRuns d(4, dynamic_base);
        const u32 n = d.size();
        const auto a = d.take(n - 100);
        const auto b = d.take(60);
        CHECK(a == 4u && b == 4 + n - 100);
        CHECK(!d.take(50));
        d.release(*a, 50);
        CHECK(d.take(50) == 4u);
        CHECK(d.take(40) == 4 + n - 40);
    }
    // host.rs dynamic_search_can_find_a_run_across_the_cursor.
    {
        DynamicRuns d(4, dynamic_base);
        const u32 n = d.size();
        const auto first = d.take(1), left = d.take(2), right = d.take(2), rest = d.take(n - 5);
        CHECK(first == 4u && left == 5u && right == 7u && rest == 9u);
        d.release(*left, 2);
        CHECK(d.take(2) == left);
        d.release(*left, 2);
        d.release(*right, 2);
        CHECK(d.take(4) == left);
        CHECK(!d.take(1));
        d.release(*first, 1);
        d.release(*first + n - 1, 1);
        CHECK(!d.take(2));
    }

    // sampler.rs console_descriptions_decode and its LOD and bounds vectors.
    {
        u32 w[27]{};
        const auto bits = [](float f) { u32 u; std::memcpy(&u, &f, 4); return u; };
        w[1] = 1;  // bilinear magnification
        w[2] = 0;  // point minification
        w[3] = 2;  // linear mips
        w[6] = w[7] = w[8] = 3;
        w[10] = bits(1000.0f);
        D3D12_SAMPLER_DESC d = console_sampler(w);
        CHECK(d.Filter == 0x5 && d.AddressW == D3D12_TEXTURE_ADDRESS_MODE_CLAMP && d.MaxLOD == 1000.0f);
        w[16] = 4;
        CHECK(console_sampler(w).Filter == 0x85 && console_sampler(w).ComparisonFunc == D3D12_COMPARISON_FUNC_LESS_EQUAL);
        w[16] = 1;
        CHECK(console_sampler(w).Filter == 0x5);
        u32 m[27]{};
        m[3] = 0;
        m[9] = bits(3.0f);
        m[10] = bits(8.0f);
        d = console_sampler(m);
        CHECK(d.MinLOD == 3.0f && d.MaxLOD == 3.0f);
        m[3] = 2;
        m[9] = bits(4.0f);
        m[10] = bits(1.0f);
        d = console_sampler(m);
        CHECK(d.MinLOD == 4.0f && d.MaxLOD == 4.0f);
        u32 b[27]{};
        b[6] = 0;
        b[7] = 6;
        b[8] = 3;
        b[11] = bits(24.0f);
        b[16] = 99;
        d = console_sampler(b);
        CHECK(d.MipLODBias == D3D12_MIP_LOD_BIAS_MAX);
        CHECK(d.AddressU == D3D12_TEXTURE_ADDRESS_MODE_CLAMP && d.AddressV == D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        CHECK(d.ComparisonFunc == D3D12_COMPARISON_FUNC_NEVER);
    }
    // The X2WP record of a placed buffer view.
    {
        PlacedRecord p;
        p.address = 0x123456789a0ull;
        p.first = 3;
        p.elements = 40;
        p.stride = 12;
        p.raw = false;
        p.unordered = true;
        p.width = 4096;
        u32 w[8];
        placed_words(p, w);
        PlacedRecord q;
        CHECK(placed_of(w, q) && q.address == p.address && q.first == 3 && q.elements == 40 && q.stride == 12 && q.unordered &&
              !q.raw && q.width == 4096);
        w[0] = 0;
        CHECK(!placed_of(w, q));
    }
    // SlotSet.
    {
        SlotSet s;
        s.resize(200);
        CHECK(!s.put(3) && s.put(3) && !s.put(130));
        CHECK(s.has(130) && !s.has(131) && !s.has(1000));
        const auto t = s.take();
        CHECK(t.size() == 2 && t[0] == 3 && t[1] == 130 && !s.has(3));
    }

    // A copied 2D descriptor can ask for all levels from its first streamed mip.
    {
        D3D12_RESOURCE_DESC r{};
        r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        r.Width = r.Height = 512; r.DepthOrArraySize = 1; r.MipLevels = 10;
        r.Format = DXGI_FORMAT_BC7_UNORM; r.SampleDesc.Count = 1;
        r.Layout = D3D12_TEXTURE_LAYOUT(0x105); r.Flags = D3D12_RESOURCE_FLAGS(0x300000);
        D3D12_SHADER_RESOURCE_VIEW_DESC v{}, decoded{};
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Format = r.Format; v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MostDetailedMip = 3;
        v.Texture2D.MipLevels = UINT32_MAX;
        for (UINT width : {512u, 2048u}) {
            auto invalid = r;
            invalid.Width = width;
            invalid.MipLevels = width == 512 ? 61 : 193;
            UINT words[8]{};
            CHECK(!x2w_late_texture::encode(invalid, v, 1, words));
            CHECK(x2w_late_texture::encode(r, v, 1, words));
            words[3] = width | (r.Height << 16);
            words[4] = (words[4] & 0xffffff) | (UINT(invalid.MipLevels) << 24);
            D3D12_RESOURCE_DESC decoded_shape{};
            CHECK(!x2w_late_texture::decode(words, decoded_shape, decoded));
            words[4] = (words[4] & 0xffff00) | (UINT(r.MipLevels) << 24);
            CHECK(!x2w_late_texture::decode(words, decoded_shape, decoded));
        }
        auto limit = r;
        limit.Width = 1; limit.Height = 1; limit.DepthOrArraySize = 2048; limit.MipLevels = 12;
        CHECK(!x2w_late_texture::valid_shape(limit)); // array count does not extend the mip chain
        limit.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        CHECK(x2w_late_texture::valid_shape(limit));  // volume depth does
        ++limit.MipLevels;
        CHECK(!x2w_late_texture::valid_shape(limit));
        limit = r; limit.Width = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION + 1;
        CHECK(!x2w_late_texture::valid_shape(limit));
        for (UINT count : {7u, 65535u, UINT32_MAX}) {
            v.Texture2D.MipLevels = count;
            UINT words[8]{};
            CHECK(x2w_late_texture::encode(r, v, (UINT64{3} << 32) + (1u << 20), words));
            D3D12_RESOURCE_DESC resource{};
            x2w_late_texture::decode(words, resource, decoded);
            CHECK(decoded.Texture2D.MostDetailedMip == 3 && decoded.Texture2D.MipLevels == count &&
                  decoded.Shader4ComponentMapping == v.Shader4ComponentMapping);
        }
    }

    // Array views spanning packed fields.
    for (const auto format : {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC7_UNORM}) {
        D3D12_RESOURCE_DESC r{};
        r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        r.Width = format == DXGI_FORMAT_BC7_UNORM ? 512 : 256; r.Height = UINT(r.Width);
        r.DepthOrArraySize = format == DXGI_FORMAT_BC1_UNORM ? 64 : 32;
        r.MipLevels = r.Width == 512 ? 10 : 9; r.Format = format;
        r.SampleDesc.Count = 1; r.Layout = D3D12_TEXTURE_LAYOUT(0x10a); r.Flags = D3D12_RESOURCE_FLAGS(0x300000);
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = format; v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        for (int partial = 0; partial < 2; ++partial) {
            if (partial) r.DepthOrArraySize = 2048;
            auto& a = v.Texture2DArray;
            a.FirstArraySlice = partial ? 1031 : 0; a.ArraySize = r.DepthOrArraySize - a.FirstArraySlice;
            a.MostDetailedMip = partial ? 2 : 0; a.MipLevels = partial ? UINT32_MAX : r.MipLevels;
            a.ResourceMinLODClamp = partial ? 1.5f : 0.f;
            constexpr UINT64 address = (UINT64{3} << 32) + (1u << 20);
            UINT words[8]{}, copied[8]{};
            CHECK(x2w_late_texture::encode(r, v, address, words));
            std::memcpy(copied, words, sizeof(copied));
            D3D12_RESOURCE_DESC got{};
            D3D12_SHADER_RESOURCE_VIEW_DESC shown{};
            x2w_late_texture::decode(copied, got, shown);
            CHECK(got.Width == r.Width && got.Height == r.Height && got.Format == r.Format &&
                  got.DepthOrArraySize == r.DepthOrArraySize && got.Layout == r.Layout && got.MipLevels == r.MipLevels);
            CHECK(shown.ViewDimension == v.ViewDimension && shown.Format == v.Format &&
                  shown.Shader4ComponentMapping == v.Shader4ComponentMapping &&
                  !std::memcmp(&shown.Texture2DArray, &a, sizeof(a)));
            CHECK(x2w_late_texture::encode(got, shown, address, copied) && !std::memcmp(words, copied, sizeof(words)));
            ++a.ArraySize;
            CHECK(!x2w_late_texture::encode(r, v, address, copied));
        }
    }

    for (const bool volume : {false, true}) {
        D3D12_RESOURCE_DESC r{};
        r.Dimension = volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        r.Width = volume ? 32u : 64u; r.Height = 64; r.DepthOrArraySize = volume ? 32 : 6;
        r.Format = volume ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_BC7_UNORM;
        r.MipLevels = 1; r.SampleDesc.Count = 1; r.Layout = D3D12_TEXTURE_LAYOUT(0x105);
        r.Flags = D3D12_RESOURCE_FLAGS(0x300000);
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = r.Format; v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.ViewDimension = volume ? D3D12_SRV_DIMENSION_TEXTURE3D : D3D12_SRV_DIMENSION_TEXTURECUBE;
        if (volume) v.Texture3D = {0, UINT32_MAX, 0.25f};
        else v.TextureCube = {0, 1, 0};
        UINT marker[8]{}, repeat[8]{};
        CHECK(x2w_late_texture::encode(r, v, 0xb6a100000, marker));
        D3D12_RESOURCE_DESC got{};
        D3D12_SHADER_RESOURCE_VIEW_DESC recovered{};
        x2w_late_texture::decode(marker, got, recovered);
        CHECK(got.Dimension == r.Dimension && got.DepthOrArraySize == r.DepthOrArraySize && got.Layout == r.Layout);
        CHECK(recovered.ViewDimension == v.ViewDimension && recovered.Shader4ComponentMapping == v.Shader4ComponentMapping &&
              !std::memcmp(&recovered.Texture3D, &v.Texture3D, sizeof(v.Texture3D)));
        CHECK(x2w_late_texture::encode(got, recovered, 0xb6a100000, repeat) && !std::memcmp(marker, repeat, sizeof(repeat)));
    }
    if (failures) {
        std::printf("startup_gpu_heap_layout_check: %d failures\n", failures);
        return 1;
    }
    std::printf("startup_gpu_heap_layout_check: pass\n");
    return 0;
}
