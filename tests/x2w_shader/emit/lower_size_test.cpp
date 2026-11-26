// Port of the upstream Rust module emit/size_tests.rs, in Rust order.
//
// End-to-end tests of what a program tells about a view by how it uses it: descriptors read for
// sizes (the dwords of a view no instruction uses) and counters, lowered to DXIL, validated by
// dxil.dll when present, and run on the host GPU and WARP.
//
// (In the Rust reference run, the_record_count_of_a_structured_buffer_read_by_index killed the
// test process without a message; see GMR/vs_port/lower/NOTES.md. Ported all the same.)

#include "lower_card_util.h"

using namespace card;

namespace {

// Lowers a wave32 compute program of one thread with the table of views at s0 and the view heap
// at s1; `srv`: the dwords and the stride of the used resource at the table's first entry (the
// third is a raw UAV). Throws ShaderError (Rust `?` on every stage).
HostShader build(std::span<const std::uint32_t> words, std::pair<std::uint8_t, std::uint32_t> srv) {
    auto param = [](RootType root, std::uint8_t user_data) { return root_param(root, 0, 1, 0, user_data, 0); };
    auto range = [](ResourceType kind, std::uint32_t table_offset_dwords, std::uint8_t dwords, std::uint32_t stride) {
        return used_resource(kind, RootType::Table(), table_offset_dwords, 0, stride, dwords, 0, 0);
    };
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = 32;
    native.layout = {param(RootType::Table(), 0), param(RootType::ResourceHeap(), 1)};
    native.used = {range(ResourceType::Srv(), 0, srv.first, srv.second), range(ResourceType::Uav(), 16, 4, 0)};
    native.io = compute_io(2 << 1, {1, 1, 1});
    native.code = le_bytes(words);
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, native.vgprs);
    vs::structure::Structured st = vs::structure::structure(p);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered lowered = vs::lower::lower(parts_of(native, p, st, io, res));
    HostShader shader = host_shader(std::move(lowered), {});
    validate(shader, true);
    return shader;
}

// Rust `build(..).is_ok()`: a ShaderError is an Err; a panic (validation) stays a failure.
bool builds(std::span<const std::uint32_t> words, std::pair<std::uint8_t, std::uint32_t> srv) {
    try {
        build(words, srv);
        return true;
    } catch (const ShaderError&) {
        return false;
    }
}

// Rust `build(..).expect("translates")`.
HostShader translates(std::span<const std::uint32_t> words, std::pair<std::uint8_t, std::uint32_t> srv) {
    try {
        return build(words, srv);
    } catch (const ShaderError& e) {
        fail("translates: " + e.to_string());
    }
}

// Rust `got[..n] == want`.
bool starts(const std::vector<std::uint32_t>& got, std::initializer_list<std::uint32_t> want) {
    return got.size() >= want.size() && std::equal(want.begin(), want.end(), got.begin());
}

// The table at s[2:3], its raw UAV's descriptor in s[20:23].
const std::uint32_t TABLE[] = {
    0xBE82'0300,  // s_mov_b32 s2, s0
    0xBE83'0380,  // s_mov_b32 s3, 0
    0xF408'0501,  // s_load_dwordx4 s[20:23], s[2:3], 64
    0xFA00'0040,
};

// Stores s4 and s5 at bytes 0 and 4 of the UAV.
const std::uint32_t STORE[] = {
    0x7E08'0204,  // v_mov_b32 v4, s4
    0x7E0A'0205,  // v_mov_b32 v5, s5
    0xE070'0000,  // buffer_store_dword v4, off, s[20:23], 0
    0x8005'0400,
    0xE070'0004,  // buffer_store_dword v5, off, s[20:23], 0 offset:4
    0x8005'0500,
    0xBF81'0000,  // s_endpgm
};

// `GetDimensions` of a texture as the console compiler expands it, from the descriptor in
// s[8:15]: width in s4, height in s5, base level in s6.
const std::uint32_t SIZES[] = {
    0x9004'9E09,  // s_lshr_b32 s4, s9, 30
    0x8F05'820A,  // s_lshl_b32 s5, s10, 2
    0x8804'0504,  // s_or_b32 s4, s4, s5
    0x8704'FF04,  // s_and_b32 s4, s4, 0xffff
    0x0000'FFFF,
    0x9385'FF0A,  // s_bfe_u32 s5, s10, 0x10000e
    0x0010'000E,
    0x9386'FF0B,  // s_bfe_u32 s6, s11, 0x4000c
    0x0004'000C,
    0x8104'8104,  // s_add_i32 s4, s4, 1
    0x8105'8105,  // s_add_i32 s5, s5, 1
    0x9004'0604,  // s_lshr_b32 s4, s4, s6
    0x9005'0605,  // s_lshr_b32 s5, s5, s6
};

// s_load_dwordx8 s[8:15], s[6:7], s12: entry 1 of the view heap.
const std::uint32_t FROM_HEAP[] = {
    0xBE86'0301,  // s_mov_b32 s6, s1
    0xBE87'0380,  // s_mov_b32 s7, 0
    0xBE8C'03A0,  // s_mov_b32 s12, 32
    0xF40C'0203,
    0x1800'0000,
};

// `s_load_dwordx4 s[8:11], s[2:3], 0`, then the record count in s4 and 0 in s5.
const std::uint32_t COUNT[] = {
    0xF408'0201,
    0xFA00'0000,
    0xBE84'030A,  // s_mov_b32 s4, s10
    0xBE85'0380,  // s_mov_b32 s5, 0
};

// ---- rs:141 -------------------------------------------------------------------------------------
// A puddle pass of the title picks three textures from the view heap, reads the first one's size
// and samples the other two.
void the_size_of_a_heap_texture_nothing_reads() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__the_size_of_a_heap_texture_nothing_reads");
    std::vector<std::uint32_t> words = concat({TABLE, FROM_HEAP, SIZES, STORE});
    HostShader shader = translates(words, {8, 0});
    const Texel one[1] = {};
    const Texel texels[15] = {};
    const View views[] = {texture(1, 1, one), texture(5, 3, texels), raw_uav(2)};
    for (auto& [label, got] : run(shader, views)) require(starts(got, {5, 3}), label);
}

// ---- rs:158 -------------------------------------------------------------------------------------
// Without the height nothing tells a 2D texture from the others: taken for one, as most are. A
// volume's depth or an array's layers are no 2D texture's.
void sizes_of_a_volume_or_an_array_stay_untranslated() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__sizes_of_a_volume_or_an_array_stay_untranslated");
    const std::uint32_t width[] = {
        0x9004'9E09,  // s_lshr_b32 s4, s9, 30
        0x8F05'820A,  // s_lshl_b32 s5, s10, 2
        0x8804'0504,  // s_or_b32 s4, s4, s5
    };
    require(builds(concat({TABLE, FROM_HEAP, width, STORE}), {8, 0}), "width alone translates");
    const std::uint32_t depth[] = {
        0x8805'0C05,  // s_or_b32 s5, s5, s12
    };
    require(!builds(concat({TABLE, FROM_HEAP, SIZES, depth, STORE}), {8, 0}), "a depth stays untranslated");
}

// ---- rs:177 -------------------------------------------------------------------------------------
// A water simulation step runs only while a structured buffer is bound: it tests the record count
// of a view it never reads.
void the_record_count_of_a_structured_buffer_nothing_reads() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__the_record_count_of_a_structured_buffer_nothing_reads");
    std::vector<std::uint32_t> words = concat({TABLE, COUNT, STORE});
    HostShader shader = translates(words, {4, 16});
    const std::uint32_t data[12] = {};
    const Texel texel[1] = {};
    // The runtime makes structured views raw.
    const View views[] = {buffer(data, stride_raw()), texture(1, 1, texel), raw_uav(2)};
    for (auto& [label, got] : run(shader, views)) require(starts(got, {3, 0}), label);
    // A stride of 4 or less may be a typed view's component count.
    require(!builds(words, {4, 4}), "stride 4 refused");
    require(!builds(words, {4, 0}), "stride 0 refused");
}

// ---- rs:201 -------------------------------------------------------------------------------------
// As the_record_count_of_a_structured_buffer_nothing_reads, of one the program also reads by
// index.
void the_record_count_of_a_structured_buffer_read_by_index() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__the_record_count_of_a_structured_buffer_read_by_index");
    const std::uint32_t load[] = {
        0x7E06'0280,  // v_mov_b32 v3, 0
        0xE030'2000,  // buffer_load_dword v5, v3, s[8:11], 0 idxen
        0x8002'0503,
        0x7E0A'0505,  // v_readfirstlane_b32 s5, v5
    };
    std::vector<std::uint32_t> words = concat({TABLE, COUNT, load, STORE});
    HostShader shader = translates(words, {4, 16});
    std::uint32_t data[12] = {};
    data[0] = 77;
    const Texel texel[1] = {};
    const View views[] = {buffer(data, stride_raw()), texture(1, 1, texel), raw_uav(2)};
    for (auto& [label, got] : run(shader, views)) require(starts(got, {3, 77}), label);
}

// ---- rs:228 -------------------------------------------------------------------------------------
// An upscaler counts its groups in a texture with atomics and sets the counter back with a store
// of one channel, 16 bits wide.
std::vector<std::uint32_t> counter_program(std::array<std::uint32_t, 2> store, std::uint32_t value) {
    const std::uint32_t words[] = {
        0xF40C'0201u,  // s_load_dwordx8 s[8:15], s[2:3], 0
        0xFA00'0000,
        0x7E00'0280,  // v_mov_b32 v0, 0
        0x7E02'0280,  // v_mov_b32 v1, 0
        0x7E04'0283,  // v_mov_b32 v2, 3
        0xF044'3108,  // image_atomic_add v2, v[0:1], s[8:15] dmask:0x1 2D glc
        0x0002'0200,
        value,  // v_mov_b32 v3, ..
        store[0],
        store[1],
        0x7E08'0281,  // v_mov_b32 v4, 1
        0xF044'3108,  // image_atomic_add v4, v[0:1], s[8:15] dmask:0x1 2D glc
        0x0002'0400,
        0xBE84'0380,  // s_mov_b32 s4, 0
        0xBE85'0380,  // s_mov_b32 s5, 0
    };
    return concat({TABLE, words, STORE});
}

void a_counter_texture_is_set_by_a_store_of_one_channel() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__a_counter_texture_is_set_by_a_store_of_one_channel");
    // image_store v3, v[0:1], s[8:15] dmask:0x1 2D, with and without d16
    const std::array<std::uint32_t, 2> stores[] = {{0xF020'1108, 0x0002'0300}, {0xF020'1108, 0x8002'0300}};
    for (const auto& store : stores) {
        HostShader shader = translates(counter_program(store, 0x7E06'0289), {8, 0});  // v3 = 9
        const Texel texel[1] = {};
        const View views[] = {uint_texture_uav(1, 1), texture(1, 1, texel), raw_uav(2)};
        for (auto& [label, got] : run(shader, views)) require(!got.empty() && got[0] == 10, label);
    }
    // Sixteen bits of a number only the card knows may be signed.
    require(!builds(counter_program({0xF020'1108, 0x8002'0300}, 0x7E06'0302), {8, 0}), "v3 = v2 refused");  // v3 = v2
    // A store of two channels says nothing about the view.
    require(!builds(counter_program({0xF020'1308, 0x0002'0300}, 0x7E06'0289), {8, 0}), "two channels refused");
}

// ---- rs:271 -------------------------------------------------------------------------------------
// A load past the wave's cache (GLC) of a buffer other groups write: its view is globally coherent
// on the host, as is every access to it.
void a_buffer_loaded_past_the_cache_is_coherent() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__a_buffer_loaded_past_the_cache_is_coherent");
    auto program = [](std::uint32_t load) {
        const std::uint32_t words[] = {
            0x7E04'028Bu,  // v_mov_b32 v2, 11
            0xE070'0008,   // buffer_store_dword v2, off, s[20:23], 0 offset:8
            0x8005'0200,
            load,  // buffer_load_dword v3, off, s[20:23], 0 offset:8
            0x8005'0300,
            0x7E08'0503,  // v_readfirstlane_b32 s4, v3
            0xBE85'0380,  // s_mov_b32 s5, 0
        };
        return translates(concat({TABLE, words, STORE}), {8, 0});
    };
    // (Rust evaluates the tuple left to right: plain first.)
    HostShader plain = program(0xE030'0008);
    HostShader coherent = program(0xE030'C008);
    // the annotations of the store's and the load's handles
    require(plain.dxil != coherent.dxil, "plain.dxil != coherent.dxil");
    const Texel texel[1] = {};
    const View views[] = {texture(1, 1, texel), texture(1, 1, texel), raw_uav(3)};
    for (const HostShader* shader : {&plain, &coherent})
        for (auto& [label, got] : run(*shader, views)) require(starts(got, {11, 0, 11}), label);
}

// ---- rs:303 -------------------------------------------------------------------------------------
// A buffer picked from the view heap and read as raw bytes is read at the host's raw twin of its
// slot, when the host keeps some; an index past the title's slots reads the last twin, not a slot
// of something else.
void a_picked_buffer_is_read_at_its_raw_twin() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__a_picked_buffer_is_read_at_its_raw_twin");
    auto program = [](std::uint32_t at) {
        const std::uint32_t words[] = {
            0xBE86'0301,  // s_mov_b32 s6, s1
            0xBE87'0380,  // s_mov_b32 s7, 0
            0xBE8C'03FF,  // s_mov_b32 s12, at
            at,
            0xF408'0203,  // s_load_dwordx4 s[8:11], s[6:7], s12
            0x1800'0000,
            0xE030'0004,  // buffer_load_dword v5, off, s[8:11], 0 offset:4
            0x8002'0500,
            0x7E0A'0505,  // v_readfirstlane_b32 s5, v5
            0xBE84'0380,  // s_mov_b32 s4, 0
        };
        return concat({TABLE, words, STORE});
    };
    const std::uint32_t title[] = {1, 2, 3, 4};
    const std::uint32_t twin[] = {5, 6, 7, 8};
    const std::uint32_t last[] = {9, 10, 11, 12};
    const Texel texel[1] = {};
    auto raw = [](std::span<const std::uint32_t> data) { return buffer(data, stride_raw()); };
    // (The store goes through the table's third entry: its twin, too.)
    auto views = [&](bool twins) {
        auto uav = [&](bool here) { return here ? raw_uav(2) : texture(1, 1, texel); };
        return std::vector<View>{
            texture(1, 1, texel), raw(title), uav(!twins), texture(1, 1, texel),
            texture(1, 1, texel), raw(twin),  uav(twins),  raw(last),
        };
    };
    const std::array<std::uint32_t, 3> cases[] = {{0, 32, 2}, {4, 32, 6}, {4, 32 * 1000, 10}};
    for (auto [first, at, expected] : cases) {
        HostShader shader;
        {
            RawViewsHere here(first);  // Rust RAW_VIEWS_HERE.set(Some(first)) ... .set(None)
            shader = translates(program(at), {8, 0});
        }
        std::vector<View> v = views(first != 0);
        for (auto& [label, got] : run(shader, v))
            require(starts(got, {0, expected}), std::string(label) + ": twins from " + std::to_string(first) +
                                                    ", byte " + std::to_string(at));
    }
}

// ---- rs:349 -------------------------------------------------------------------------------------
// A table's buffer read as raw bytes is read at its raw twin too.
void a_tables_raw_buffer_is_read_at_its_raw_twin() {
    x2w::vs::lower::test::set_dump_name("emit__size_tests__a_tables_raw_buffer_is_read_at_its_raw_twin");
    const std::uint32_t load[] = {
        0xF408'0201u,  // s_load_dwordx4 s[8:11], s[2:3], 0
        0xFA00'0000,
        0xE030'0004,  // buffer_load_dword v5, off, s[8:11], 0 offset:4
        0x8002'0500,
        0x7E0A'0505,  // v_readfirstlane_b32 s5, v5
        0xBE84'0380,  // s_mov_b32 s4, 0
    };
    std::vector<std::uint32_t> words = concat({TABLE, load, STORE});
    const std::uint32_t title[] = {1, 2, 3, 4};
    const std::uint32_t twin[] = {5, 6, 7, 8};
    const Texel texel[1] = {};
    HostShader shader;
    {
        RawViewsHere here(4);
        shader = translates(words, {4, 0});
    }
    auto other = [&] { return texture(1, 1, texel); };
    const View views[] = {buffer(title, stride_raw()), other(), other(), other(), buffer(twin, stride_raw()),
                          other(), raw_uav(2)};
    for (auto& [label, got] : run(shader, views)) require(starts(got, {0, 6}), label);
}

}  // namespace

int main(int argc, char** argv) {
    // `--skip <name>` leaves a test out (as cargo test's --skip; the Rust run skipped
    // the_record_count_of_a_structured_buffer_read_by_index, which ended the process there).
    std::vector<std::string_view> skip;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == "--skip") skip.push_back(argv[++i]);
    auto guarded = [&](const char* name, void (*f)()) {
        if (std::find(skip.begin(), skip.end(), std::string_view(name)) != skip.end()) {
            std::fprintf(stderr, "%s: skipped\n", name);
            return;
        }
        card::guarded(name, f);
    };
    guarded("the_size_of_a_heap_texture_nothing_reads", the_size_of_a_heap_texture_nothing_reads);
    guarded("sizes_of_a_volume_or_an_array_stay_untranslated", sizes_of_a_volume_or_an_array_stay_untranslated);
    guarded("the_record_count_of_a_structured_buffer_nothing_reads",
            the_record_count_of_a_structured_buffer_nothing_reads);
    guarded("the_record_count_of_a_structured_buffer_read_by_index",
            the_record_count_of_a_structured_buffer_read_by_index);
    guarded("a_counter_texture_is_set_by_a_store_of_one_channel", a_counter_texture_is_set_by_a_store_of_one_channel);
    guarded("a_buffer_loaded_past_the_cache_is_coherent", a_buffer_loaded_past_the_cache_is_coherent);
    guarded("a_picked_buffer_is_read_at_its_raw_twin", a_picked_buffer_is_read_at_its_raw_twin);
    guarded("a_tables_raw_buffer_is_read_at_its_raw_twin", a_tables_raw_buffer_is_read_at_its_raw_twin);
    return test_util::finish("lower_size_test");
}
