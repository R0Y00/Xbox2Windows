// Port of the upstream Rust shader tests in src/bvh.rs.
#include "x2w_shader/bvh.h"

#include "vs_front_test.h"

#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>

using namespace x2w::vs::bvh;
using namespace x2w::vs::bvh::testing;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();
const float NaN = std::numeric_limits<float>::quiet_NaN();

constexpr Box UNIT{-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

Box moved(Box b, Vec3 by)
{
    Box out{};
    for (int k = 0; k < 6; ++k) out[k] = b[k] + by[k % 3];
    return out;
}

Box off() { return {NaN, NaN, NaN, NaN, NaN, NaN}; }

Ray with_extent(Ray r, float extent)
{
    r.extent = extent;
    return r;
}

struct Hit {
    float t, i, j;
    bool front;
};

// The hit a result describes: distance, I, J, front face.
std::optional<Hit> hit(Result4 r)
{
    const float t = std::bit_cast<float>(r[0]), det = std::bit_cast<float>(r[1]);
    const float i = std::bit_cast<float>(r[2]), j = std::bit_cast<float>(r[3]);
    if (t == INF) return std::nullopt;
    return Hit{t / det, i / det, j / det, det > 0.0f};
}

// Clockwise seen from the origin looking down +z in a left-handed system (x
// right, y up): the front face.
const Triangle TRIANGLE{{{0.0f, 2.0f, 5.0f}, {2.0f, -2.0f, 5.0f}, {-2.0f, -2.0f, 5.0f}}};

Words tri_node(const Triangle& t, std::uint32_t id) { return triangles(t.data(), 3, id); }

const Result4 MISS_BITS = to_bits(MISS);

}  // namespace

// An empty slot's inverted infinite bounds do not come before the children a
// ray enters.
VS_TEST(an_empty_slot_is_never_entered)
{
    auto inside = [](float x) { return Box{x - 1.0f, -1.0f, -1.0f, x + 1.0f, 1.0f, 1.0f}; };
    const float up = INF, down = -INF;
    const Box empty{up, up, up, down, down, down};
    const Boxes bounds{inside(30.0f), empty, inside(10.0f), empty};
    const Ray ray = Ray::make({0, 0, 0}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((boxes({8, NONE, 16, NONE}, bounds, ray, true) == Result4{16, 8, NONE, NONE}));
}

VS_TEST(node_pointers_name_64_byte_nodes_past_the_descriptors_base)
{
    const std::uint64_t structure = 0x12'3456'7800ull;
    const std::uint64_t pointer = ((structure >> 8) << 5) + ((0x40 >> 3) | 5);
    VS_CHECK(kind(pointer) == 5);
    VS_CHECK(node_address(0, BOX_SORT, pointer) == structure + 0x40);
    const auto word0 = static_cast<std::uint32_t>(structure >> 8);
    const auto word1 = static_cast<std::uint32_t>(structure >> 40) | BOX_SORT;
    VS_CHECK(node_address(word0, word1, (0x1000 >> 3) | 4) == structure + 0x1000);
    VS_CHECK(node_address(0, 0x8000'0001u, 8) == (1ull << 40) + 64);
}

VS_TEST(halves_convert_exactly)
{
    VS_CHECK(half_to_f32(0x3c00) == 1.0f);
    VS_CHECK(half_to_f32(0xc000) == -2.0f);
    VS_CHECK(half_to_f32(0x7c00) == INF);
    VS_CHECK(half_to_f32(0x0001) == std::ldexp(1.0f, -24));
    VS_CHECK(half_to_f32(0x03ff) == 1023.0f * std::ldexp(1.0f, -24));
    VS_CHECK(half_to_f32(0x7bff) == 65504.0f);
    VS_CHECK(std::isnan(half_to_f32(0x7e00)));
    VS_CHECK(std::bit_cast<std::uint32_t>(half_to_f32(0x8000)) == 0x8000'0000u);
    VS_CHECK(half_to_f32(f32_to_half(0.1f, true)) >= 0.1f);
    VS_CHECK(half_to_f32(f32_to_half(0.1f, false)) <= 0.1f);
}

VS_TEST(boxes_are_returned_nearest_first_and_misses_last)
{
    const Result4 kids{0x105, 0x20c, 0x304, 0x400};
    const Words node = box32(kids, {moved(UNIT, {30.0f, 0.0f, 0.0f}), moved(UNIT, {10.0f, 0.0f, 0.0f}),
                                    moved(UNIT, {10.0f, 5.0f, 0.0f}), moved(UNIT, {20.0f, 0.0f, 0.0f})});
    const Ray ray = Ray::make({0, 0, 0}, {1.0f, 0.0f, 0.0f}, 1000.0f);
    VS_CHECK((intersect(node, 5, ray, true) == Result4{0x20c, 0x400, 0x105, NONE}));
    VS_CHECK((intersect(node, 5, ray, false) == Result4{0x105, 0x20c, NONE, 0x400}));
    VS_CHECK((intersect(node, 5, with_extent(ray, 19.5f), true) == Result4{0x20c, 0x400, NONE, NONE}));
    VS_CHECK((intersect(node, 5, with_extent(ray, 19.0f), true) == Result4{0x20c, NONE, NONE, NONE}));
    const Ray back = Ray::make({0, 0, 0}, {-1.0f, 0.0f, 0.0f}, 1000.0f);
    VS_CHECK((intersect(node, 5, back, true) == Result4{NONE, NONE, NONE, NONE}));
}

VS_TEST(a_ray_starting_inside_a_box_enters_it_at_its_origin)
{
    const Words node = box32({8, 16, NONE, NONE}, {UNIT, moved(UNIT, {3.0f, 0.0f, 0.0f}), off(), off()});
    const Ray ray = Ray::make({0, 0, 0}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((intersect(node, 5, ray, true) == Result4{8, 16, NONE, NONE}));
    VS_CHECK((intersect(node, 5, with_extent(ray, 1e-6f), true) == Result4{8, NONE, NONE, NONE}));
}

VS_TEST(axis_aligned_rays_and_origins_on_a_face_make_no_nan)
{
    const Words node = box32({8, 16, 24, 32}, {UNIT, off(), off(), off()});
    const Ray ray = Ray::make({-5.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK(std::isinf(ray.inverse[1]));
    VS_CHECK((intersect(node, 5, ray, true) == Result4{8, NONE, NONE, NONE}));
    const Ray top = Ray::make({-5.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((intersect(node, 5, top, true) == Result4{NONE, NONE, NONE, NONE}));
    const Ray outside = Ray::make({-5.0f, 1.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((intersect(node, 5, outside, true) == Result4{NONE, NONE, NONE, NONE}));
    const Ray minus = Ray::make({-5.0f, 0.5f, 0.0f}, {1.0f, -0.0f, 0.0f}, 100.0f);
    VS_CHECK(minus.inverse[1] == -INF);
    VS_CHECK((intersect(node, 5, minus, true) == Result4{8, NONE, NONE, NONE}));
}

VS_TEST(a_box_switched_off_and_a_missing_child_return_none)
{
    const Words node = box32({8, NONE, 24, 32}, {UNIT, UNIT, Box{NaN, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f}, off()});
    const Ray ray = Ray::make({-5.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((intersect(node, 5, ray, true) == Result4{8, NONE, NONE, NONE}));
}

VS_TEST(equal_distances_keep_child_order)
{
    const Words node = box32({8, 16, 24, 32}, {UNIT, UNIT, UNIT, UNIT});
    const Ray ray = Ray::make({-5.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 100.0f);
    VS_CHECK((intersect(node, 5, ray, true) == Result4{8, 16, 24, 32}));
}

VS_TEST(half_float_boxes_test_like_their_widened_values)
{
    const Boxes bounds{moved(UNIT, {30.0f, 0.1f, 0.0f}), moved(UNIT, {10.0f, 0.0f, 0.3f}),
                       moved(UNIT, {10.0f, 5.0f, 0.0f}), moved(UNIT, {20.7f, 0.0f, 0.0f})};
    const Result4 kids{0x105, 0x20c, 0x304, 0x400};
    const Words node = box16(kids, bounds);
    const Boxes wide = boxes16(node);
    for (int b = 0; b < 4; ++b)
        for (int k = 0; k < 3; ++k) VS_CHECK(wide[b][k] <= bounds[b][k] && wide[b][3 + k] >= bounds[b][3 + k]);
    const Ray ray = Ray::make({0, 0, 0}, {1.0f, 0.0f, 0.0f}, 1000.0f);
    VS_CHECK((intersect(node, 4, ray, true) == Result4{0x20c, 0x400, 0x105, NONE}));
    VS_CHECK(intersect(node, 4, ray, true) == intersect(box32(kids, wide), 5, ray, true));
}

VS_TEST(a_triangle_returns_distance_and_weights_over_one_denominator)
{
    const Words node = tri_node(TRIANGLE, 9);
    auto h = hit(intersect(node, 0, Ray::make({0, 0, 0}, {0.0f, 0.0f, 1.0f}, 100.0f), true));
    VS_REQUIRE(h.has_value());
    VS_CHECK(h->t == 5.0f);
    VS_CHECK(h->i == 0.25f && h->j == 0.25f);
    VS_CHECK(h->front);
    h = hit(intersect(node, 0, Ray::make({1.75f, -1.75f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f), true));
    VS_REQUIRE(h.has_value());
    VS_CHECK(h->i > 0.85f && h->j < 0.1f);
    h = hit(intersect(node, 0, Ray::make({0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, -1.0f}, 100.0f), true));
    VS_REQUIRE(h.has_value());
    VS_CHECK(h->t == 5.0f && h->i == 0.25f && h->j == 0.25f && !h->front);
    h = hit(intersect(node, 0, Ray::make({0, 0, 0}, {0.0f, 0.0f, 4.0f}, 100.0f), true));
    VS_REQUIRE(h.has_value());
    VS_CHECK(h->t == 1.25f);
}

VS_TEST(a_triangle_is_missed_beside_behind_and_edge_on)
{
    const Words node = tri_node(TRIANGLE, 9);
    VS_CHECK(intersect(node, 0, Ray::make({3.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f), true) == MISS_BITS);
    VS_CHECK(intersect(node, 0, Ray::make({0.0f, 0.0f, 6.0f}, {0.0f, 0.0f, 1.0f}, 100.0f), true) == MISS_BITS);
    VS_CHECK(intersect(node, 0, Ray::make({-5.0f, 0.0f, 5.0f}, {1.0f, 0.0f, 0.0f}, 100.0f), true) == MISS_BITS);
    const Words flat = tri_node({{{0.0f, 0.0f, 5.0f}, {1.0f, 1.0f, 5.0f}, {2.0f, 2.0f, 5.0f}}}, 9);
    VS_CHECK(intersect(flat, 0, Ray::make({1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f), true) == MISS_BITS);
    const Words nan = tri_node({{{NaN, 2.0f, 5.0f}, TRIANGLE[1], TRIANGLE[2]}}, 9);
    VS_CHECK(intersect(nan, 0, Ray::make({0, 0, 0}, {0.0f, 0.0f, 1.0f}, 100.0f), true) == MISS_BITS);
    VS_CHECK(intersect(node, 0, Ray::make({0, 0, 0}, {0, 0, 0}, 100.0f), true) == MISS_BITS);
    VS_CHECK(hit(intersect(node, 0, Ray::make({0, 0, 0}, {0.0f, 0.0f, 1.0f}, 1.0f), true)).has_value());
}

VS_TEST(a_ray_through_a_shared_edge_misses_neither_triangle)
{
    const Vec3 quad[4] = {{-1.0f, -1.0f, 4.0f}, {-1.0f, 1.0f, 4.0f}, {1.0f, 1.0f, 4.0f}, {1.0f, -1.0f, 4.0f}};
    const Words first = tri_node({{quad[0], quad[1], quad[2]}}, 9);
    const Words second = tri_node({{quad[0], quad[2], quad[3]}}, 9);
    const Vec3 dirs[3] = {{0.0f, 0.0f, 1.0f}, {0.125f, -0.0625f, 1.0f}, {0.25f, 0.5f, 2.0f}};
    for (int n = -40; n <= 40; ++n) {
        const float x = static_cast<float>(n) / 64.0f;
        const std::pair<float, float> origins[3] = {
            {x, x}, {x, x + 1.0f / 1024.0f}, {x + 1.0f / 1024.0f, x}};
        for (auto [ox, oy] : origins) {
            for (const Vec3& d : dirs) {
                const float s = 4.0f / d[2];
                const Ray ray = Ray::make({ox - d[0] * s, oy - d[1] * s, 0.0f}, d, 100.0f);
                const auto a = hit(intersect(first, 0, ray, true));
                const auto b = hit(intersect(second, 0, ray, true));
                VS_CHECK(a.has_value() || b.has_value());
                for (const auto& h : {a, b}) {
                    if (h) VS_CHECK(std::fabs(h->t - s) < 1e-5f && h->front);
                }
            }
        }
    }
}

VS_TEST(the_id_dword_says_which_vertices_the_weights_belong_to)
{
    const Vec3 a = TRIANGLE[0], b = TRIANGLE[1], c = TRIANGLE[2];
    const Ray ray = Ray::make({0.5f, -0.25f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f);
    const Result4 plain = intersect(tri_node({{a, b, c}}, 9), 0, ray, true);
    const std::pair<std::uint32_t, std::uint32_t> ids[6] = {{1, 2}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {1, 0}};
    for (auto [i, j] : ids) {
        Triangle stored{};
        stored[3 - i - j] = a;
        stored[i] = b;
        stored[j] = c;
        VS_CHECK(intersect(tri_node(stored, i | (j << 2)), 0, ray, true) == plain);
    }
    VS_CHECK(intersect(tri_node({{a, b, c}}, 9 | 0x80), 0, ray, true) == plain);
}

VS_TEST(triangles_one_to_three_take_their_vertices_and_their_id_byte)
{
    const Vec3 a = TRIANGLE[0], b = TRIANGLE[1], c = TRIANGLE[2];
    const Vec3 far_{50.0f, 50.0f, 50.0f};
    const Ray ray = Ray::make({0.5f, -0.25f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f);
    const Result4 plain = intersect(tri_node({{a, b, c}}, 9), 0, ray, true);
    for (std::uint32_t k = 1; k < 4; ++k) {
        Vec3 vertices[5] = {far_, far_, far_, far_, far_};
        const auto& at = TRIANGLE_VERTICES[k];
        vertices[at[0]] = a;
        vertices[at[1]] = b;
        vertices[at[2]] = c;
        const Words node = triangles(vertices, 5, 9u << (8 * k));
        VS_CHECK(intersect(node, k, ray, true) == plain);
    }
    const Vec3 d{2.0f, 2.0f, 5.0f};
    const Vec3 four[4] = {a, b, c, d};
    const Words node = triangles(four, 4, 9 | ((1 | (2 << 2)) << 8));
    VS_CHECK((triangle_vertices(node, 1) == Triangle{{b, d, c}}));
}

VS_TEST(instances_and_procedural_nodes_return_a_triangle_miss)
{
    const Words node = tri_node(TRIANGLE, 9);
    const Ray ray = Ray::make({0, 0, 0}, {0.0f, 0.0f, 1.0f}, 100.0f);
    for (std::uint32_t k : {6u, 7u}) VS_CHECK(intersect(node, k, ray, true) == MISS_BITS);
}

VS_TEST(the_largest_axis_and_its_sign_do_not_change_the_answer)
{
    auto turn = [](Vec3 v, int axis, bool flip) {
        Vec3 r = v;
        for (int n = 0; n < axis; ++n) r = {r[2], r[0], r[1]};
        if (flip) {
            const int largest = (2 + axis) % 3;
            r[largest] = -r[largest];
            r[(largest + 1) % 3] = -r[(largest + 1) % 3];
        }
        return r;
    };
    const Ray ray = Ray::make({0.25f, -0.5f, 0.0f}, {0.0625f, 0.125f, 1.0f}, 100.0f);
    const auto base = hit(to_bits(triangle(TRIANGLE, ray)));
    VS_REQUIRE(base.has_value());
    for (int axis = 0; axis < 3; ++axis) {
        for (bool flip : {false, true}) {
            Triangle p{};
            for (int v = 0; v < 3; ++v) p[v] = turn(TRIANGLE[v], axis, flip);
            const Ray r = Ray::make(turn(ray.origin, axis, flip), turn(ray.direction, axis, flip), 100.0f);
            const auto h = hit(to_bits(triangle(p, r)));
            VS_REQUIRE(h.has_value());
            VS_CHECK(std::fabs(h->t - base->t) < 1e-5f);
            VS_CHECK(std::fabs(h->i - base->i) < 1e-5f && std::fabs(h->j - base->j) < 1e-5f);
            VS_CHECK(h->front == base->front);
        }
    }
}

VS_TEST_MAIN()
