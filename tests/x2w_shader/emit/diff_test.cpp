// Port of the upstream Rust module emit/diff_tests.rs. Differential tests: HLSL compiled by the console's shader
// compiler and translated, against the same HLSL compiled for PC, both drawn (or dispatched) on
// the host GPU with the same inputs.
//
// Needs X2W_SHADERCFG_CONSOLE_DXC (the console SDK's dxc.exe) and a PC dxc.exe (X2W_SHADERCFG_PC_DXC, else the
// newest Windows SDK's); skipped without them. Set X2W_SHADERCFG_WARP=1 to draw on WARP.
// X2W_SHADERCFG_DIFF_ONLY=<case name> runs one case; X2W_SHADERCFG_EMIT_DUMP=<dir> writes the console and
// translated containers there.
//
// Each Rust #[test] is a function here; a Rust panic (unwrap, expect, assert) is an exception
// that fails that function (main catches it) and the others still run.

#include <windows.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/host.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"

namespace fs = std::filesystem;
using namespace x2w::vs;
using host::Dispatch;
using host::Draw;
using host::Host;
using host::Root;
using host::Stride;
using host::View;
using validator::Validator;

namespace {

struct Case {
    const char* name;
    const char* src;
};

struct MeshCase {
    const char* name;
    std::uint32_t groups;
    const char* src;
};

// Pixel shaders, each `ps` (CASES). Compute shaders, each `cs`, dispatched as 2x2 groups of 8x8
// threads over a 16x16 texture and a 256-dword buffer (CS_CASES). Mesh shaders, each `ms` (the
// console runs one thread per output lane, so groups of 32 run as 64 lanes there), with their
// group counts, drawn with `mps` into 16x16 targets (MS_CASES). The 2020 compiler's containers
// do not record the thread count, so cases reading the group id keep max(threads, vertices,
// primitives) at the lane count (see emit).
const char* const RS = R"hlsl(RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0)), DescriptorTable(Sampler(s0)), StaticSampler(s1, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_WRAP, addressV=TEXTURE_ADDRESS_MIRROR), StaticSampler(s2, filter=FILTER_MIN_MAG_LINEAR_MIP_POINT, addressU=TEXTURE_ADDRESS_BORDER, addressV=TEXTURE_ADDRESS_CLAMP, borderColor=STATIC_BORDER_COLOR_OPAQUE_WHITE, mipLODBias=0.5))hlsl";
const char* const COMMON = R"hlsl(
cbuffer K : register(b0) { float4 k; };
Texture2D tex : register(t0);
SamplerState smp : register(s0);
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0; };
[RootSignature(RS)]
V vs(float4 p : POSITION, float2 uv : TEXCOORD0) {
    V o;
    o.pos = p;
    o.uv = uv;
    o.col = float4(uv * k.xy, p.x * k.z, 1);
    return o;
}
)hlsl";
const char* const CS_RS = R"hlsl(RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0, numDescriptors=4)), DescriptorTable(Sampler(s0)), DescriptorTable(UAV(u0, numDescriptors=4)), CBV(b1), SRV(t4), DescriptorTable(SRV(t0, space=2, numDescriptors=9)), StaticSampler(s3, filter=FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, comparisonFunc=COMPARISON_LESS_EQUAL, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP))hlsl";
const char* const CS_COMMON = R"hlsl(
cbuffer K : register(b0) { float4 k; };
Texture2D tex : register(t0);
Buffer<float4> typed : register(t1);
struct S { float a; uint b; float2 c; };
StructuredBuffer<S> structured : register(t2);
ByteAddressBuffer raw : register(t3);
SamplerState smp : register(s0);
SamplerComparisonState cmp : register(s3);
RWByteAddressBuffer outb : register(u0);
RWTexture2D<float4> outt : register(u1);
cbuffer R : register(b1) { float4 table[16]; };
ByteAddressBuffer rootraw : register(t4);
// `raw`'s bytes in typed views (the formats the console's typed loads name).
Buffer<float4> unorm8x4 : register(t0, space2);
Buffer<float4> snorm16x2 : register(t1, space2);
Buffer<uint4> uint8x4 : register(t2, space2);
Buffer<float4> unorm10x3_2 : register(t3, space2);
Buffer<uint4> uint16x4 : register(t4, space2);
Buffer<float> unorm8 : register(t5, space2);
Buffer<float> half16 : register(t6, space2);
Buffer<float4> snorm8x4 : register(t7, space2);
RWTexture2D<uint> ut : register(u2);
TextureCube cubet : register(t8, space2);
#define FMT(numeric, data) ((numeric) << 4 | (data))
#define CS [RootSignature(RS)] [numthreads(8, 8, 1)] \
    void cs(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID, \
            uint3 d : SV_DispatchThreadID, uint i : SV_GroupIndex)
)hlsl";
const char* const MS_COMMON = R"hlsl(
#define RS "RootConstants(num32BitConstants=4, b0)"
cbuffer K : register(b0) { float4 k; };
struct V { float4 pos : SV_Position; float4 col : COLOR0; float2 uv : TEXCOORD0; };
[RootSignature(RS)] float4 mps(V i) : SV_Target { return i.col + float4(i.uv, 0, 0); }
#define MS_ATTR [RootSignature(RS)] [outputtopology("triangle")]
)hlsl";

const Case CASES[] = {
    {"constant", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target { return float4(0.25, 0.5, 1, 0.75); })hlsl"},
    {"interpolated", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target { return i.col * 2 - float4(i.uv, 0, 0); })hlsl"},
    {"sample and branch", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            float4 t = tex.Sample(smp, i.uv);
            float4 r = t * i.col + k.w;
            if (i.uv.x > 0.5) r.xy = sin(r.yx * 3.0);
            else r.z = rcp(r.z + 2.0);
            return r;
        })hlsl"},
    // A wave-wide range scalarises a per-pixel loop (as the title's light
    // lists do): the console's 64-lane sections become host wave operations;
    // the result is the same for any wave the pixel lands in.
    {"wave range", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint idx = uint(i.pos.x * 0.37 + i.pos.y * 0.21) & 15;
            uint lo = WaveActiveMin(idx), hi = WaveActiveMax(idx);
            float s = 0;
            for (uint n = lo; n <= hi; n++) if (n == idx) s += n * k.x;
            return float4(s, lo <= idx, hi >= idx, WaveActiveCountBits(true) > 0);
        })hlsl"},
    {"uniform loop", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            float s = 0;
            for (uint n = 0; n < (asuint(k.x) & 7); n++) s += i.uv.y * n;
            return float4(s, 0, 0, 1);
        })hlsl"},
    {"bit loop", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            uint s = p.y;
            for (uint n = 0; n < (p.x & 7); n++) s = (s ^ (s << 3)) + n;
            return float4(s & 255, 0, 0, 1);
        })hlsl"},
    {"integer loop", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            uint s = 0;
            for (uint n = 0; n < (p.x & 7); n++) s += n;
            return float4(s, 0, 0, 1);
        })hlsl"},
    {"loop", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float s = 0;
            for (uint n = 0; n < (p.x & 7); n++) s += i.uv.y * n + k.x;
            return float4(s, 0, 0, 1);
        })hlsl"},
    {"discard", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            if ((p.x ^ p.y) & 1) discard;
            return i.col;
        })hlsl"},
    {"immediate samplers", R"hlsl(SamplerState point_wrap : register(s1);
        SamplerState linear_border : register(s2);
        [RootSignature(RS)] float4 ps(V i) : SV_Target {
            return tex.Sample(point_wrap, i.uv * 3 - 1) + 2 * tex.Sample(linear_border, i.uv * 2.5 - 0.7);
        })hlsl"},
    {"loop, discard and loads", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            if ((p.x ^ p.y) & 1) discard;
            float s = 0;
            for (uint n = 0; n < (p.x & 7); n++) s += i.uv.y * n + k.x;
            float4 t = tex.Load(int3(p & 3, 0));
            return float4(s, frac(i.uv.x * 7.3), max(t.x, t.y) * k.y, sqrt(abs(i.col.x)) + exp2(i.uv.y));
        })hlsl"},
    {"level of detail", R"hlsl([RootSignature(RS)] float4 ps(V i) : SV_Target {
            return float4(tex.CalculateLevelOfDetail(smp, i.uv * 4),
                tex.CalculateLevelOfDetailUnclamped(smp, i.uv * 7), 0, 1);
        })hlsl"},
    // 16-bit outputs and inputs (one value per dword with the 2020
    // compiler; the title's two per dword cannot be compiled
    // here) and the front face.
    {"16-bit varyings and front face", R"hlsl(struct V16 { float4 pos : SV_Position; half2 uv : TEXCOORD0; float4 col : COLOR0; half4 h : TEXCOORD1; };
        [RootSignature(RS)] V16 vs16(float4 p : POSITION, float2 uv : TEXCOORD0) {
            V16 o;
            o.pos = p;
            o.uv = half2(uv * k.xy);
            o.col = float4(uv, p.x * k.z, 1);
            o.h = half4(uv.yx * 3 - 1, k.w, p.y);
            return o;
        }
        [RootSignature(RS)] float4 ps(V16 i, bool f : SV_IsFrontFace) : SV_Target {
            float4 c = i.col * float4(i.uv, i.h.xy) + float4(i.h.zw, 0, 0);
            return f ? c : -c;
        })hlsl"},
};

const Case CS_CASES[] = {
    {"ids and stores", R"hlsl(CS {
            outb.Store(4 * (d.y * 16 + d.x), asuint(d.x * k.x + d.y));
            outt[d.xy] = float4(g.x, g.y, t.x, t.y + i);
        })hlsl"},
    {"shared memory and barrier", R"hlsl(groupshared float lds[64];
        groupshared uint cnt;
        CS {
            if (i == 0) cnt = 0;
            lds[i] = d.x * k.x + d.y;
            GroupMemoryBarrierWithGroupSync();
            InterlockedAdd(cnt, i);
            float v = lds[63 - i] + lds[(i + 9) & 63];
            GroupMemoryBarrierWithGroupSync();
            outb.Store(4 * (d.y * 16 + d.x), asuint(v));
            outt[d.xy] = float4(v, cnt, 0, 1);
        })hlsl"},
    {"sample and divergent store", R"hlsl(CS {
            float4 s = tex.SampleLevel(smp, (d.xy + 0.5) / 16.0, 0);
            if ((d.x ^ d.y) & 1) outt[d.xy] = s * k.y;
            else outb.Store(4 * (d.y * 16 + d.x), asuint(s.x + s.w));
        })hlsl"},
    {"buffers and root views", R"hlsl(CS {
            float4 a = typed[i & 15];
            S s = structured[(i * 3) & 7];
            uint r = raw.Load(4 * (i & 31));
            float4 c = table[d.x & 15];
            uint q = rootraw.Load(4 * ((i + 5) & 31));
            outb.Store(4 * (d.y * 16 + d.x), asuint(a.x + a.w * s.a) ^ s.b ^ r ^ q);
            outt[d.xy] = c + float4(s.c, a.yz);
        })hlsl"},
    // (The 2020 compiler branches around every narrowed region, so these
    // become branches, not guards.)
    {"conditional memory access", R"hlsl(CS {
            float v = d.x * k.x + d.y;
            if (i & 1) v = v * 3 + 1;
            if (i & 4) v += tex.Load(int3(d.xy & 3, 0)).y;
            if (i & 2) outb.Store(4 * (d.y * 16 + d.x), asuint(v));
            if (i & 8) outt[d.xy] = float4(v, 0, 0, 1);
        })hlsl"},
    // Each side of a per-group branch leaves a compare in VCC; the select
    // after the join reads the joined mask (as `postfx_splines` does).
    {"select on joined compares", R"hlsl(CS {
            float v = d.x * k.x + d.y;
            float r;
            bool ok;
            [branch] if ((g.x ^ g.y) & 1) {
                r = sqrt(v) * k.z + sin(v * k.w);
                ok = r < 5;
            } else {
                r = v * k.y + 1;
                ok = r > -4;
            }
            float o = ok ? r : -1;
            outb.Store(4 * (d.y * 16 + d.x), asuint(o));
            outt[d.xy] = float4(o, r, v, 1);
        })hlsl"},
    // One host wave of 32 lanes holds the group, as the console's wave of 64
    // does, so lane numbers agree with the PC compile.
    {"lane reads", R"hlsl([RootSignature(RS)] [numthreads(8, 4, 1)]
        void cs(uint3 d : SV_DispatchThreadID) {
            float v = d.x * k.x + d.y;
            float w = WaveReadLaneAt(v, 5);
            uint lane = WaveGetLaneIndex();
            uint q = QuadReadAcrossX(lane * 10) + QuadReadAcrossDiagonal(lane);
            outb.Store(4 * (d.y * 16 + d.x), q);
            outt[d.xy] = float4(v, w, QuadReadAcrossY(v), lane);
        })hlsl"},
    // Bytes and halves: SDWA operand and result selection, byte permutes,
    // f16 conversions.
    {"sub-dword operations", R"hlsl(CS {
            uint n = d.y * 16 + d.x;
            uint x = raw.Load(4 * (n & 31)), y = raw.Load(4 * ((n * 7) & 31));
            float a = float(x >> 16) + float((y >> 8) & 255) * k.x;
            float b = f16tof32(x >> 16) + f16tof32(y);
            uint p = (x & 0xff) | ((y & 0xff) << 8) | (x & 0xff0000) | ((y >> 8) << 24);
            uint q = f32tof16(a) | (f32tof16(b) << 16);
            int s = int(x << 24) >> 24;
            uint u = (x & 0xffff) + (y >> 16) + ((x >> 24) ^ (y & 0xffff)) + asuint(k.w) * 1000 + 77;
            outb.Store4(16 * n, uint4(asuint(a), asuint(b) ^ p, q, asuint(float(s)) + u));
        })hlsl"},
    // 64-bit additions (VALU carries as lane masks) and explicit gradients.
    {"carries and gradients", R"hlsl(CS {
            uint n = d.y * 16 + d.x;
            uint64_t a = (uint64_t(raw.Load(4 * (n & 31))) << 32) | raw.Load(4 * ((n * 3) & 31));
            uint64_t b = (uint64_t(raw.Load(4 * ((n * 5) & 31))) << 32) | raw.Load(4 * ((n * 7) & 31));
            uint64_t s = a + b, u = a - b;
            float4 gr = tex.SampleGrad(smp, (d.xy + 0.5) / 16.0, float2(k.x, 0) * 0.1, float2(0, k.z) * 0.1);
            outb.Store4(16 * n, uint4(uint(s), uint(s >> 32) ^ uint(u), uint(u >> 32), asuint(gr.x + gr.y * 3)));
        })hlsl"},
    // Comparison sampling at level 0 with an immediate comparison sampler.
    {"comparison sampling", R"hlsl(CS {
            float r = tex.SampleCmpLevelZero(cmp, (d.xy + 0.25) / 15.0, k.x * d.x / 16.0);
            outb.Store(4 * (d.y * 16 + d.x), asuint(r));
        })hlsl"},
    // 16-bit texture results, two channels per register (`d16`).
    {"16-bit texture reads", R"hlsl(Texture2D<half4> htex : register(t0);
        CS {
            half4 a = htex.SampleLevel(smp, (d.xy + 0.5) / 16.0, 0);
            half3 b = htex.Load(int3(d.xy & 3, 0)).xyz;
            outb.Store4(16 * (d.y * 16 + d.x), uint4(asuint(float(a.x) + a.w),
                asuint(float(a.y) * a.z), asuint(float(b.x)), asuint(float(b.y) - b.z)));
        })hlsl"},
    // Sizes: the console compiler computes them from descriptor bits
    // (width, height, levels, record counts).
    {"sizes from descriptors", R"hlsl(CS {
            uint w, h, l, n, b, ns, st;
            tex.GetDimensions(0, w, h, l);
            typed.GetDimensions(n);
            raw.GetDimensions(b);
            structured.GetDimensions(ns, st);
            // Read too: the translation takes the views' kinds from those reads.
            float r = tex.Load(int3(d.xy & 3, 0)).x + typed[i & 15].y + structured[i & 7].a;
            outb.Store4(16 * (d.y * 16 + d.x),
                uint4(w * 7 + h, l + asuint(r), n * 5 + ns * 3 + st, b + raw.Load(4 * (i & 31))));
        })hlsl"},
    // Typed loads (`tbuffer_load_format_*`, the title's vertex
    // fetch) convert by the format the instruction names: the PC reads the
    // same bytes through typed views in those formats. Channels past the
    // format's read (0, 0, 0, 1); a byte format at unaligned offsets.
    {"typed loads", R"hlsl(CS {
            uint n = d.y * 16 + d.x, e = n & 31, f = n >> 5;
            uint4 a, b, c, dd, ee, ff, gg, hh;
        #ifdef CONSOLE
            a = raw.XB_TypedLoad4(4 * e, FMT(0, 10));
            b = raw.XB_TypedLoad4(4 * e, FMT(1, 5));
            c = raw.XB_TypedLoad4(4 * e, FMT(4, 10));
            dd = raw.XB_TypedLoad4(4 * e, FMT(0, 9));
            ee = raw.XB_TypedLoad4(8 * (e & 15), FMT(4, 12));
            ff = uint4(raw.XB_TypedLoad((3 * e) & 127, FMT(0, 1)), 0, 0, 0);
            gg = uint4(raw.XB_TypedLoad2(2 * e, FMT(7, 2)), 0, 0);
            hh = raw.XB_TypedLoad4(4 * e, FMT(1, 10));
        #else
            a = asuint(unorm8x4[e]);
            b = asuint(snorm16x2[e]);
            c = uint8x4[e];
            dd = asuint(unorm10x3_2[e]);
            ee = uint16x4[e & 15];
            ff = uint4(asuint(unorm8[(3 * e) & 127]), 0, 0, 0);
            gg = uint4(asuint(half16[e]), 0, 0, 0);
            hh = asuint(snorm8x4[e]);
        #endif
            uint4 v = f == 0 ? a : f == 1 ? b : f == 2 ? c : f == 3 ? dd
                : f == 4 ? ee : f == 5 ? ff : f == 6 ? gg : hh;
            outb.Store4(16 * n, v);
        })hlsl"},
    // Wave operations: the console expands them into whole-wave sections
    // (DPP, `v_permlanex16`, lane reads) and lane masks read as numbers
    // (`s_bcnt1`, `s_ff1`, `v_mbcnt`). 32 threads: one wave on the console
    // (lanes 32-63 inactive) and on the host; the branch leaves lanes out.
    {"wave operations", R"hlsl([RootSignature(RS)] [numthreads(8, 4, 1)]
        void cs(uint3 d : SV_DispatchThreadID, uint i : SV_GroupIndex) {
            uint n = d.y * 16 + d.x;
            float v = d.x * k.x + d.y;
            uint u = raw.Load(4 * (i & 31));
            uint4 o = 0;
            if (u & 3) {
                o.x = WaveActiveSum(u >> 8) + WaveActiveCountBits((u & 4) != 0);
                o.y = asuint(WaveActiveMax(v)) ^ WaveActiveMin(u) ^ WaveActiveBitOr(u)
                    ^ WaveActiveBitAnd(u | 1) ^ asuint(WaveActiveMin(v)) ^ WaveActiveMax(int(u));
                o.z = WavePrefixSum(u & 255) + WavePrefixCountBits(v > 2) * 1000;
                // (The PC compile evaluates a wave operation in a `?:` arm
                // in that arm's lanes only, so it stays outside.)
                o.w = asuint(WaveActiveSum(v)) + (WaveIsFirstLane() ? WaveReadLaneFirst(u) : 0);
            }
            outb.Store4(16 * n, o);
        })hlsl"},
    // Atomics the console aggregates per wave (one add of the lane count,
    // each lane's offset from `v_mbcnt`), and texture atomics. Results
    // that depend on the lanes' order are left out.
    {"aggregated and texture atomics", R"hlsl(CS {
            uint u = raw.Load(4 * (i & 31)), r, q;
            outb.InterlockedAdd(0, 1, r);
            outb.InterlockedAdd(4 * (1 + r % 8), 1);
            if (u & 1) outb.InterlockedAdd(64, u & 255);
            uint2 p = d.xy & 3;
            InterlockedMax(ut[p], u);
            InterlockedOr(ut[p + uint2(4, 0)], u >> (i & 7));
            InterlockedMin(ut[p + uint2(8, 0)], u);
            InterlockedAdd(ut[p + uint2(12, 0)], u & 15, q);
            InterlockedXor(ut[p + uint2(0, 4)], u);
        })hlsl"},
    // A loop whose count comes from a view that reads zeros (as a null or
    // not yet mirrored one does) runs four billion times on the console;
    // the translation leaves it after `LOOP_LIMIT` iterations. Run on WARP
    // only (`X2W_SHADERCFG_WARP=1`), ours only: without the limit it would hang
    // the GPU.
    {"terminates on zero data", R"hlsl(CS {
            uint cnt = outb.Load(4 * 1000), s = d.x;
            for (uint j = 0; j != cnt - 1; j++) s = s * 3 + raw.Load(4 * (j & 31));
            outb.Store(4 * (d.y * 16 + d.x), s);
        })hlsl"},
    // Cube sampling: the console computes face coordinates (`v_cube*`)
    // and samples the face; the host samples the direction rebuilt from
    // them. Every face and texel: a major axis of +-2 and the others at
    // texel centres, so both sides compute them exactly.
    {"cube sampling", R"hlsl(CS {
            float sg = (i & 8) ? -2 : 2;
            float a = (d.x & 3) - 1.5, b = (d.y & 3) - 1.5;
            uint ax = i % 3;
            float3 dir = ax == 0 ? float3(sg, a, b) : ax == 1 ? float3(a, sg, b) : float3(a, b, sg);
            float4 p = cubet.SampleLevel(smp, dir, 0);
            float4 q = cubet.SampleLevel(smp, -dir.zxy, 0);
            outb.Store4(16 * (d.y * 16 + d.x), asuint(p + q * 3));
        })hlsl"},
    // A local array indexed per lane (M0-relative VGPR moves), bit fields
    // and first bits from the top.
    {"indexed registers and bit fields", R"hlsl(CS {
            uint n = d.y * 16 + d.x;
            uint x = raw.Load(4 * (n & 31)), y = raw.Load(4 * ((n * 5) & 31));
            uint arr[24];
            [unroll] for (uint j = 0; j < 24; j++) arr[j] = x ^ (j * 0x01010101);
            [loop] for (uint m = 0; m < (y & 3); m++) arr[(y >> (m * 5)) % 24] = y + m;
            uint v = arr[(x >> 3) % 24];
            uint w = asuint(k.x) & 31, o = n & 15;
            uint bf = (x & ((1u << w) - 1)) << o;
            uint fb = firstbithigh(asuint(k.y)) + firstbithigh(x) * 3 + firstbithigh(int(y)) * 7;
            outb.Store4(16 * n, uint4(v, bf, fb, arr[n % 24]));
        })hlsl"},
    // Size queries at a level known per lane (`image_get_resinfo`).
    {"size queries", R"hlsl(CS {
            uint w, h, l, cw, ch, cl;
            tex.GetDimensions(d.x >> 4, w, h, l);
            cubet.GetDimensions(d.y >> 4, cw, ch, cl);
            float r = tex.Load(int3(d.xy & 3, 0)).x;
            outb.Store4(16 * (d.y * 16 + d.x), uint4(w * 7 + h, l + asuint(r), cw * 3 + ch, cl));
        })hlsl"},
    // Gathers with offsets computed per lane (programmable offsets).
    {"gather with offsets", R"hlsl(CS {
            int2 o = int2(int(d.x & 7) - 4, int(d.y % 5) - 2);
            float4 r = tex.GatherRed(smp, (d.xy + 0.5) / 16.0, o);
            float4 q = tex.GatherGreen(smp, (d.yx + 0.25) / 16.0, -o);
            outb.Store4(16 * (d.y * 16 + d.x), asuint(r + q * 3));
        })hlsl"},
    // Constant arrays: the console compiler appends them to the code and
    // reads them through `s_getpc_b64` (scalar loads for a uniform index,
    // global loads per lane, 16-bit ones too).
    {"constant tables", R"hlsl(static const uint tbl[16] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
        static const float ftbl[8] = { 0.5, -1, 2.25, 3, -4.5, 6, 7.75, -8 };
        // (16-bit types: compiled with half precision enabled.)
        static const uint16_t stbl[12] = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 65535 };
        CS {
            uint n = d.y * 16 + d.x;
            uint u = asuint(k.x) & 15;
            uint a = tbl[i & 15] ^ tbl[u] << 8;
            float f = ftbl[(i * 3) & 7] * ftbl[u & 7];
            uint h = stbl[(n * 5) % 12] + stbl[u % 12];
            outb.Store4(16 * n, uint4(a, asuint(f), h, tbl[(n + u) & 15]));
        })hlsl"},
    // 16-bit arithmetic (scalar and packed), 16-bit compares, typed stores
    // of halves (`buffer_store_format_d16`), 64-bit shifts and compares.
    {"16-bit and 64-bit arithmetic", R"hlsl(RWBuffer<half4> hb : register(u3);
        CS {
            uint n = d.y * 16 + d.x;
            uint x = raw.Load(4 * (n & 31)), y = raw.Load(4 * ((n * 7) & 31));
            half a = half(d.x * k.x - 2), b = half(d.y + k.y), c = half(k.z);
            half2 p = half2(a, b) * half2(b, c) + half2(c, a);
            half m = max(a, b) - min(a, c) * a;
            bool lt = a < b, ge = p.x >= p.y;
            hb[n] = half4(p.x, m, lt ? a : b, ge ? p.y : a - b);
            uint64_t q = (uint64_t(x) << (n & 31)) | y, w = uint64_t(y) << 3;
            uint z = q < w ? 1 : q == w ? 2 : 3;
            outb.Store2(8 * n, uint2(uint(q >> 7) ^ z, uint(w >> (n & 15))));
        })hlsl"},
};

const MeshCase MS_CASES[] = {
    {"one triangle", 1, R"hlsl(MS_ATTR [numthreads(32, 1, 1)]
        void ms(uint i : SV_GroupIndex, out vertices V verts[3], out indices uint3 tris[1]) {
            SetMeshOutputCounts(3, 1);
            if (i < 3) {
                verts[i].pos = float4(i == 1 ? 3 : -1, i == 2 ? 3 : -1, 0.5, 1);
                verts[i].col = float4(k.x * i, k.y, k.z * (i == 2), 1);
                verts[i].uv = float2(i * 0.5, k.w);
            }
            if (i == 0) tris[0] = uint3(0, 1, 2);
        })hlsl"},
    // A quad per group in its quadrant; a run-time vertex count, and a
    // primitive count per group whose third triangle is culled.
    {"groups, counts and culling", 4, R"hlsl(struct P { bool cull : SV_CullPrimitive; };
        MS_ATTR [numthreads(64, 1, 1)]
        void ms(uint i : SV_GroupIndex, uint3 g : SV_GroupID, out vertices V verts[8],
                out indices uint3 tris[4], out primitives P prims[4]) {
            uint nv = 4 + (k.z > 1);
            uint np = 2 + (g.x & 1);
            SetMeshOutputCounts(nv, np);
            float2 o = float2(g.x & 1, g.x >> 1) - 1;
            if (i < nv) {
                float2 p = i < 4 ? float2(i & 1, i >> 1) : float2(-0.75, 0.5);
                verts[i].pos = float4(o + p, 0.5, 1);
                verts[i].col = float4(k.x * i, g.x * 0.25, k.z, 1);
                verts[i].uv = float2(i, g.x);
            }
            if (i < np) {
                tris[i] = i == 0 ? uint3(0, 1, 2) : i == 1 ? uint3(1, 3, 2) : uint3(1, 4, 3);
                prims[i].cull = i == 2;
            }
        })hlsl"},
    // Two waves per group, as the title's mesh programs run: the
    // second wave's lanes are threads 64-127.
    {"two waves", 1, R"hlsl(MS_ATTR [numthreads(128, 1, 1)]
        void ms(uint i : SV_GroupIndex, out vertices V verts[81], out indices uint3 tris[128]) {
            SetMeshOutputCounts(81, 128);
            if (i < 81) {
                float2 p = float2(i % 9, i / 9) / 8.0;
                verts[i].pos = float4(p * 2 - 1, 0.5, 1);
                verts[i].col = float4(p.x * k.x, p.y, (i & 7) * 0.125, 1);
                verts[i].uv = float2(i, k.y);
            }
            uint c = i / 2 % 8 + i / 16 * 9;
            tris[i] = i & 1 ? uint3(c + 1, c + 10, c + 9) : uint3(c, c + 1, c + 9);
        })hlsl"},
    // Quad reads (DPP quad permutations on the console; the title's
    // tessellating mesh programs swizzle within quads too).
    {"quad reads", 1, R"hlsl(MS_ATTR [numthreads(128, 1, 1)]
        void ms(uint i : SV_GroupIndex, out vertices V verts[81], out indices uint3 tris[128]) {
            SetMeshOutputCounts(81, 128);
            float w = i * k.x + k.y;
            float a = QuadReadLaneAt(w, 0);
            float b = QuadReadLaneAt(w, 2);
            float c = QuadReadAcrossX(w);
            if (i < 81) {
                float2 p = float2(i % 9, i / 9) / 8.0;
                verts[i].pos = float4(p * 2 - 1, 0.5, 1);
                verts[i].col = float4(a, b, c, 1);
                verts[i].uv = float2(i, k.y);
            }
            uint cc = i / 2 % 8 + i / 16 * 9;
            tris[i] = i & 1 ? uint3(cc + 1, cc + 10, cc + 9) : uint3(cc, cc + 1, cc + 9);
        })hlsl"},
    // Vertices and indices written by other threads than their lanes: the
    // console passes them through shared memory, with a loop and barriers.
    {"grid through shared memory", 2, R"hlsl(MS_ATTR [numthreads(32, 1, 1)]
        void ms(uint i : SV_GroupIndex, uint3 g : SV_GroupID, out vertices V verts[36],
                out indices uint3 tris[64]) {
            SetMeshOutputCounts(36, 50);
            for (uint v = i; v < 36; v += 32) {
                float2 p = float2(v % 6, v / 6) / 5.0;
                verts[v].pos = float4(p.x - g.x, p.y * 2 - 1, 0.5, 1);
                verts[v].col = float4(p * k.x, k.z * g.x, 1);
                verts[v].uv = float2(v, k.y);
            }
            if (i < 25) {
                uint c = i % 5 + i / 5 * 6;
                tris[2 * i] = uint3(c, c + 1, c + 6);
                tris[2 * i + 1] = uint3(c + 1, c + 7, c + 6);
            }
        })hlsl"},
};

// A Rust panic: fails the test function.
[[noreturn]] void panic(const std::string& what) { throw std::logic_error(what); }

std::optional<std::string> env(const char* name) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return std::nullopt;
}

bool is_warp() { return env("X2W_SHADERCFG_WARP") == std::optional<std::string>("1"); }

// Rust's `{}` (Display) of an f32: the shortest decimal that reads back, never an exponent.
std::string display(float x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
    char b[128];
    auto r = std::to_chars(b, b + sizeof b, x, std::chars_format::fixed);
    return std::string(b, r.ptr);
}

// Rust's `{:?}` (Debug) of an f32: as Display with ".0" on whole numbers, in exponent form
// below 1e-4 and from 1e16.
std::string f32_debug(float x) {
    if (std::isnan(x) || std::isinf(x)) return display(x);
    const float a = std::fabs(x);
    if (a != 0.0f && (a < 1e-4f || a >= 1e16f)) {
        char b[64];
        auto r = std::to_chars(b, b + sizeof b, x, std::chars_format::scientific);
        std::string s(b, r.ptr);
        // "1.5e-05" -> "1.5e-5", "1e+30" -> "1e30".
        const std::size_t e = s.find('e');
        std::string mant = s.substr(0, e), ex = s.substr(e + 1);
        bool neg = !ex.empty() && ex[0] == '-';
        if (!ex.empty() && (ex[0] == '-' || ex[0] == '+')) ex.erase(0, 1);
        while (ex.size() > 1 && ex[0] == '0') ex.erase(0, 1);
        return mant + "e" + (neg ? "-" : "") + ex;
    }
    std::string s = display(x);
    if (s.find('.') == std::string::npos) s += ".0";
    return s;
}

// Rust's `{:.2e}` of an f32.
std::string rust_exp(float x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
    char b[64];
    std::snprintf(b, sizeof b, "%.2e", double(x));
    std::string s(b);
    const std::size_t e = s.find('e');
    std::string mant = s.substr(0, e), ex = s.substr(e + 1);
    bool neg = ex[0] == '-';
    ex.erase(0, 1);
    while (ex.size() > 1 && ex[0] == '0') ex.erase(0, 1);
    return mant + "e" + (neg ? "-" : "") + ex;
}

std::string hex(std::uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%x", v);
    return b;
}

// f32::total_cmp's key: -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN.
std::int32_t total_key(float x) {
    std::int32_t v = std::bit_cast<std::int32_t>(x);
    v ^= std::int32_t(std::uint32_t(v >> 31) >> 1);
    return v;
}

// f32::max(1.0): NaN gives the other.
float max1(float x) { return std::isnan(x) ? 1.0f : std::max(x, 1.0f); }

std::optional<std::pair<fs::path, fs::path>> tools() {
    const std::optional<std::string> c = env("X2W_SHADERCFG_CONSOLE_DXC");
    if (!c) return std::nullopt;
    const fs::path console(*c);
    fs::path pc;
    if (const std::optional<std::string> p = env("X2W_SHADERCFG_PC_DXC")) {
        pc = *p;
    } else {
        std::optional<fs::path> d = Validator::default_path();
        if (!d) return std::nullopt;
        pc = d->replace_filename("dxc.exe");
    }
    if (fs::exists(console) && fs::exists(pc)) return std::pair{console, pc};
    return std::nullopt;
}

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) panic("cannot read " + p.string());
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& p, std::span<const std::uint8_t> bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out) panic("cannot write " + p.string());
}

std::vector<std::uint8_t> compile(const fs::path& dxc, const fs::path& dir, std::string_view src, std::string_view entry,
                                  const char* profile) {
    const fs::path input = dir / "t.hlsl";
    const fs::path output = dir / (std::string(entry) + ".bin");
    const fs::path log = dir / "dxc.log";
    std::string text;
    if (entry == "cs") {
        text = std::string("#define RS \"") + CS_RS + "\"\n" + CS_COMMON + "\n" + std::string(src) + "\n";
    } else if (entry.starts_with('m')) {
        text = std::string(MS_COMMON) + "\n" + std::string(src) + "\n";
    } else {
        text = std::string("#define RS \"") + RS + "\"\n" + COMMON + "\n" + std::string(src) + "\n";
    }
    write_file(input, std::span(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
    std::string cmd = "\"\"" + dxc.string() + "\" -T " + profile + " -E " + std::string(entry) + " -Fo \"" +
                      output.string() + "\" \"" + input.string() + "\"";
    if (src.find("half") != std::string_view::npos) cmd += " -enable-16bit-types";
    // `CONSOLE` marks the console compile (cases spell console-only intrinsics there).
    if (const std::optional<std::string> c = env("X2W_SHADERCFG_CONSOLE_DXC"); c && fs::path(*c) == dxc) cmd += " -DCONSOLE";
    // The whole command quoted once more: cmd.exe strips the outer pair. Output captured, as
    // Rust's Command::output.
    cmd += " > \"" + log.string() + "\" 2>&1\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::ifstream in(log, std::ios::binary);
        std::string err((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        panic(err);
    }
    return read_file(output);
}

// Root arguments for a translated stage: user data from the root values.
std::vector<std::uint32_t> user_data(const HostShader& s, const std::array<std::uint32_t, 4>& k) {
    std::vector<std::uint32_t> data(std::max<std::size_t>(s.user_data_dwords(), 1), 0);
    using Kd = native::RootType::Kind;
    for (const UserData& u : s.user_data) {
        std::uint32_t v;
        if (u.root.kind == Kd::Constants && u.slot == 0) {
            v = k.at(u.dword);
        } else if (u.root.kind == Kd::Table && u.slot == 1) {
            // The texture and sampler tables start at heap index 1, the UAV table (compute) at
            // 5, in the console's units.
            v = 32;
        } else if (u.root.kind == Kd::Table && u.slot == 2) {
            v = 16;
        } else if (u.root.kind == Kd::Table && u.slot == 3) {
            v = 5 * 32;
        } else if (u.root.kind == Kd::Table && u.slot == 6) {
            // The typed views of `raw` (compute) from 9.
            v = 9 * 32;
        } else {
            panic("unexpected user data (root kind " + std::to_string(int(u.root.kind)) + ", other " +
                  std::to_string(u.root.other) + ", slot " + std::to_string(u.slot) + ")");
        }
        data.at(u.index) = v;
    }
    return data;
}

// Rust's `name.replace([' ', ','], "_")`.
std::string stem_of(const char* name) {
    std::string s(name);
    for (char& c : s)
        if (c == ' ' || c == ',') c = '_';
    return s;
}

// Validates a copy of dxil when the validator is there; a rejection panics.
void validate_if_there(const std::vector<std::uint8_t>& dxil, const char* name) {
    if (std::optional<Validator> v = Validator::load(Validator::default_path())) {
        std::vector<std::uint8_t> copy = dxil;
        if (std::optional<std::string> e = v->validate(copy)) panic(std::string(name) + ": " + *e);
    }
}

host::ComPtr<ID3D12RootSignature> pc_root(const Host& host, const std::vector<std::uint8_t>& container) {
    // The PC container carries its root signature.
    host::ComPtr<ID3D12RootSignature> root;
    const HRESULT hr = host.device->CreateRootSignature(0, container.data(), container.size(), IID_PPV_ARGS(&root));
    if (FAILED(hr)) throw host::Error(host::message(hr));
    return root;
}

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

const std::array<std::uint32_t, 4> K = {bits(0.75f), bits(-1.5f), bits(2.0f), bits(0.125f)};

// (relative difference, index, ours, theirs) of the worst texel channel: Rust's max_by with
// total_cmp (NaN highest, the last of equal ones).
struct Worst {
    float rel;
    std::size_t at;
    float a, b;
    std::string debug() const {
        return "(" + f32_debug(rel) + ", " + std::to_string(at) + ", " + f32_debug(a) + ", " + f32_debug(b) + ")";
    }
};

Worst worst_of(const std::vector<std::array<float, 4>>& ours, const std::vector<std::array<float, 4>>& theirs) {
    std::optional<Worst> w;
    const std::size_t n = std::min(ours.size(), theirs.size()) * 4;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = ours[i / 4][i % 4], b = theirs[i / 4][i % 4];
        const float rel = std::fabs(a - b) / max1(std::fabs(b));
        if (!w || total_key(rel) >= total_key(w->rel)) w = Worst{rel, i, a, b};
    }
    if (!w) panic("no texels");
    return *w;
}

bool selected(const std::optional<std::string>& only, const char* name) { return !only || *only == name; }

void report(const std::vector<std::string>& failures) {
    for (const std::string& f : failures) std::fprintf(stderr, "  %s\n", f.c_str());
    CHECK(failures.empty());
}

void translated_shaders_draw_as_pc_ones() {
    const auto t = tools();
    if (!t) {
        std::fprintf(stderr, "no console or PC compiler; skipped\n");
        return;
    }
    const auto& [console, pc] = *t;
    std::optional<Host> h;
    try {
        h.emplace(false);
    } catch (const host::Error& e) {
        std::fprintf(stderr, "no SM 6.6 device (%s); skipped\n", e.what());
        return;
    }
    const Host& host = *h;
    const fs::path dir = fs::temp_directory_path() / ("gpu-diff-" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    std::vector<std::array<float, 4>> texels;
    for (int i = 0; i < 16; ++i)
        texels.push_back({float(i) / 16.0f, float(i % 4) * 0.3f, 1.0f - float(i) / 20.0f, 0.5f});
    const std::array<float, 4> dummy[] = {{9.0f, 9.0f, 9.0f, 9.0f}};
    auto sampler = [](D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address) {
        D3D12_SAMPLER_DESC d{};
        d.Filter = filter;
        d.AddressU = address;
        d.AddressV = address;
        d.AddressW = address;
        d.MaxAnisotropy = 1;
        d.MaxLOD = std::numeric_limits<float>::max();
        return d;
    };
    const D3D12_SAMPLER_DESC samplers[] = {
        sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
        sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP),
    };
    // A full-screen triangle: POSITION (4 floats), TEXCOORD0 (2 floats).
    const float vertices[] = {
        -1.0f, -1.0f, 0.5f, 1.0f, 0.0f, 1.0f,   //
        -1.0f, 3.0f,  0.5f, 1.0f, 0.0f, -1.0f,  //
        3.0f,  -1.0f, 0.5f, 1.0f, 2.0f, 1.0f,
    };
    const host::Texture textures[] = {{1, 1, dummy}, {4, 4, texels}};
    auto draw = [&](std::vector<std::pair<std::uint32_t, Root>> roots) {
        Draw d;
        d.roots = std::move(roots);
        d.textures = textures;
        d.samplers = samplers;
        d.vertices = vertices;
        d.stride = 24;
        d.size = {16, 16};
        d.mesh_groups = 0;
        return d;
    };
    std::vector<std::string> failures;
    const std::optional<std::string> only = env("X2W_SHADERCFG_DIFF_ONLY");
    for (const Case& c : CASES) {
        if (!selected(only, c.name)) continue;
        const char* name = c.name;
        const std::string_view src = c.src;
        // A case may bring its own vertex shader.
        const char* vs = src.find(" vs16(") != std::string_view::npos ? "vs16" : "vs";
        const HostShader vs_c = translate(compile(console, dir, src, vs, "vs_6_5"));
        HostShader ps_c;
        try {
            ps_c = translate(compile(console, dir, src, "ps", "ps_6_5"));
        } catch (const ShaderError& e) {
            failures.push_back(std::string(name) + ": " + e.what());
            continue;
        }
        if (const std::optional<std::string> out = env("X2W_SHADERCFG_EMIT_DUMP")) {
            const std::string stem = stem_of(name);
            write_file(fs::path(*out) / (stem + ".ps.cdxil"), compile(console, dir, src, "ps", "ps_6_5"));
            write_file(fs::path(*out) / (stem + ".ps.dxil"), ps_c.dxil);
        }
        if (std::optional<Validator> v = Validator::load(Validator::default_path())) {
            for (const HostShader* s : std::array<const HostShader*, 2>{&vs_c, &ps_c}) {
                std::vector<std::uint8_t> copy = s->dxil;
                if (std::optional<std::string> e = v->validate(copy)) panic(std::string(name) + ": " + *e);
            }
        }
        const auto [root_t, pso_t] = host.pipeline(vs_c, ps_c);
        std::vector<std::pair<std::uint32_t, Root>> roots_t;
        roots_t.emplace_back(0, Root::constants(user_data(vs_c, K)));
        roots_t.emplace_back(1, Root::constants(user_data(ps_c, K)));
        const auto ours = host.draw(root_t.Get(), pso_t.Get(), draw(std::move(roots_t)));

        const std::vector<std::uint8_t> vs_p = compile(pc, dir, src, vs, "vs_6_5");
        const std::vector<std::uint8_t> ps_p = compile(pc, dir, src, "ps", "ps_6_5");
        const auto root_p = pc_root(host, ps_p);
        const auto pso_p = host.pipeline_with(root_p.Get(), vs_p, ps_p);
        std::vector<std::pair<std::uint32_t, Root>> roots_p;
        roots_p.emplace_back(0, Root::constants(std::vector<std::uint32_t>(K.begin(), K.end())));
        roots_p.emplace_back(1, Root::table(false, 1));
        roots_p.emplace_back(2, Root::table(true, 1));
        const auto theirs = host.draw(root_p.Get(), pso_p.Get(), draw(std::move(roots_p)));

        const Worst worst = worst_of(ours, theirs);
        std::fprintf(stderr, "%s: worst relative difference %s (at %zu: %s vs %s)\n", name, rust_exp(worst.rel).c_str(),
                     worst.at, display(worst.a).c_str(), display(worst.b).c_str());
        if (std::isnan(worst.rel) || worst.rel > 1e-3f) failures.push_back(std::string(name) + ": " + worst.debug());
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    report(failures);
}

void translated_mesh_shaders_draw_as_pc_ones() {
    const auto t = tools();
    if (!t) {
        std::fprintf(stderr, "no console or PC compiler; skipped\n");
        return;
    }
    const auto& [console, pc] = *t;
    std::optional<Host> h;
    try {
        h.emplace(false);
    } catch (const host::Error& e) {
        std::fprintf(stderr, "no SM 6.6 device (%s); skipped\n", e.what());
        return;
    }
    const Host& host = *h;
    const fs::path dir = fs::temp_directory_path() / ("gpu-diff-ms-" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    auto draw = [&](std::vector<std::pair<std::uint32_t, Root>> roots, std::uint32_t mesh_groups) {
        Draw d;
        d.roots = std::move(roots);
        d.stride = 0;
        d.size = {16, 16};
        d.mesh_groups = mesh_groups;
        return d;
    };
    std::vector<std::string> failures;
    const std::optional<std::string> only = env("X2W_SHADERCFG_DIFF_ONLY");
    for (const MeshCase& c : MS_CASES) {
        if (!selected(only, c.name)) continue;
        const char* name = c.name;
        const std::uint32_t groups = c.groups;
        const std::string_view src = c.src;
        const std::vector<std::uint8_t> console_ms = compile(console, dir, src, "ms", "ms_6_5");
        HostShader ms_c;
        try {
            ms_c = translate(console_ms);
        } catch (const ShaderError& e) {
            failures.push_back(std::string(name) + ": " + e.what());
            continue;
        }
        const HostShader ps_c = translate(compile(console, dir, src, "mps", "ps_6_5"));
        if (const std::optional<std::string> out = env("X2W_SHADERCFG_EMIT_DUMP")) {
            const std::string stem = stem_of(name);
            write_file(fs::path(*out) / (stem + ".ms.cdxil"), console_ms);
            write_file(fs::path(*out) / (stem + ".ms.dxil"), ms_c.dxil);
        }
        validate_if_there(ms_c.dxil, name);
        const auto [root_t, pso_t] = host.mesh_pipeline(ms_c, ps_c);
        std::vector<std::pair<std::uint32_t, Root>> roots_t;
        roots_t.emplace_back(0, Root::constants(user_data(ms_c, K)));
        roots_t.emplace_back(1, Root::constants(user_data(ps_c, K)));
        const auto ours = host.draw(root_t.Get(), pso_t.Get(), draw(std::move(roots_t), groups));
        const std::vector<std::uint8_t> ms_p = compile(pc, dir, src, "ms", "ms_6_5");
        const std::vector<std::uint8_t> ps_p = compile(pc, dir, src, "mps", "ps_6_5");
        const auto root_p = pc_root(host, ms_p);
        const auto pso_p = host.mesh_pipeline_with(root_p.Get(), ms_p, ps_p);
        std::vector<std::pair<std::uint32_t, Root>> roots_p;
        roots_p.emplace_back(0, Root::constants(std::vector<std::uint32_t>(K.begin(), K.end())));
        const auto theirs = host.draw(root_p.Get(), pso_p.Get(), draw(std::move(roots_p), groups));
        const Worst worst = worst_of(ours, theirs);
        const std::size_t drawn = std::size_t(std::count_if(theirs.begin(), theirs.end(), [](const auto& x) { return x[3] != 0.0f; }));
        std::fprintf(stderr, "%s: %zu texels drawn; worst relative difference %s (at %zu: %s vs %s)\n", name, drawn,
                     rust_exp(worst.rel).c_str(), worst.at, display(worst.a).c_str(), display(worst.b).c_str());
        if (std::isnan(worst.rel) || worst.rel > 1e-3f || drawn == 0) {
            // Which texels each drew (alpha set), row by row.
            auto row = [](const std::vector<std::array<float, 4>>& v, std::size_t from) {
                std::string s;
                for (std::size_t i = from; i < from + 16 && i < v.size(); ++i) s += v[i][3] != 0.0f ? '#' : '.';
                return s;
            };
            for (std::size_t r = 0; r < ours.size() && r < theirs.size(); r += 16)
                std::fprintf(stderr, "  %s  %s\n", row(ours, r).c_str(), row(theirs, r).c_str());
            failures.push_back(std::string(name) + ": " + worst.debug());
        }
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    report(failures);
}

// Where PC code and the console differ by design, the translation must give the console's
// results: float-to-int conversions saturate and give 0 for NaN (v_cvt_i32/u32_f32), the legacy
// multiply the compiler uses for `*` gives 0 for 0 x NaN, `frac` never reaches 1.0 (it gives the
// largest value below; some hosts' own `frac` gives 1.0), and sine and cosine take turns: the
// sine of a half turn and the cosine of a quarter turn are exactly +0 (the host's functions of
// the rounded angle give about 1e-7). The compiler takes whole turns off itself, so large angles
// cannot be tried from here.
void float_semantics_are_the_consoles() {
    const auto t = tools();
    if (!t) {
        std::fprintf(stderr, "no console compiler; skipped\n");
        return;
    }
    const fs::path& console = t->first;
    std::optional<Host> h;
    try {
        h.emplace(false);
    } catch (const host::Error& e) {
        std::fprintf(stderr, "no SM 6.6 device (%s); skipped\n", e.what());
        return;
    }
    const Host& host = *h;
    const fs::path dir = fs::temp_directory_path() / ("gpu-float-" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    const char* src = R"hlsl([RootSignature(RS)] [numthreads(4, 1, 1)]
        void cs(uint i : SV_GroupIndex) {
            float x = i == 0 ? k.x : i == 1 ? -k.x : i == 2 ? k.y : k.z;
            outb.Store(4 * i, asuint(int(x)));
            outb.Store(16 + 4 * i, uint(x));
            outb.Store(32 + 4 * i, asuint(x * k.w));
            float y = i == 0 ? k.w - 1e-10 : i == 1 ? k.z : i == 2 ? k.w - 1e-8 : k.z + 0.25;
            outb.Store(48 + 4 * i, asuint(frac(y)));
            float r = i == 0 ? k.w + 3.14159274 : i == 1 ? k.w + 1.0
                    : i == 2 ? k.w - 3.14159274 : k.w + 1.57079637;
            outb.Store(64 + 4 * i, asuint(i == 3 ? cos(r) : sin(r)));
        })hlsl";
    const HostShader cs = translate(compile(console, dir, src, "cs", "cs_6_5"));
    const auto [root, pso] = host.compute_pipeline(cs);
    // 1e30, NaN, 3.5, 0.
    const std::array<std::uint32_t, 4> k = {bits(1e30f), 0x7FC0'0000u, bits(3.5f), 0};
    const std::array<float, 4> zero[] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    const View views[] = {
        View::texture(1, 1, zero), View::texture(1, 1, zero), View::texture(1, 1, zero),
        View::texture(1, 1, zero), View::texture(1, 1, zero), View::raw_uav(20),
        View::texture_uav(1, 1),
    };
    Dispatch d;
    d.roots.emplace_back(0, Root::constants(user_data(cs, k)));
    d.views = views;
    d.groups = {1, 1, 1};
    const auto out = host.dispatch(root.Get(), pso.Get(), d);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (out.empty() || out[0].size() < 20) panic("float_semantics_are_the_consoles: short output");
    const std::vector<std::uint32_t> want = {
        0x7FFF'FFFFu, 0x8000'0000u, 0, 3,                                  // int
        0xFFFF'FFFFu, 0, 0, 3,                                             // uint
        0, 0, 0, 0,                                                        // x * 0, NaN included
        0x3F7F'FFFFu, bits(0.5f), 0x3F7F'FFFFu, bits(0.75f),               // frac
    };
    const std::vector<std::uint32_t> got(out[0].begin(), out[0].begin() + 16);
    if (got != want) {
        for (std::size_t i = 0; i < 16; ++i)
            if (got[i] != want[i]) std::fprintf(stderr, "word %zu: %s, want %s\n", i, hex(got[i]).c_str(), hex(want[i]).c_str());
    }
    CHECK(got == want);
    // sin(pi), sin(-pi) and cos(pi/2): the compiler turns radians into exactly 0.5, -0.5 and 0.25
    // turns.
    CHECK(out[0][16] == 0 && out[0][18] == 0 && out[0][19] == 0);
    // An ordinary angle is untouched: sin(1 rad).
    const float one = std::bit_cast<float>(out[0][17]);
    if (!(std::fabs(one - 0.841'470'96f) < 1e-4f)) std::fprintf(stderr, "sin(1) = %s\n", display(one).c_str());
    CHECK(std::fabs(one - 0.841'470'96f) < 1e-4f);
}

// Equal bits, both zero (the console's legacy multiply gives +0 where PC code gives -0), or two
// normal floats within 1e-3 relative.
bool close_bits(std::uint32_t a, std::uint32_t b) {
    const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
    return a == b || (x == 0.0f && y == 0.0f) ||
           (std::isnormal(x) && std::isnormal(y) && std::fabs(x - y) <= 1e-3f * std::max(std::fabs(y), 1.0f));
}

void translated_compute_matches_pc() {
    const auto t = tools();
    if (!t) {
        std::fprintf(stderr, "no console or PC compiler; skipped\n");
        return;
    }
    const auto& [console, pc] = *t;
    std::optional<Host> h;
    try {
        h.emplace(false);
    } catch (const host::Error& e) {
        std::fprintf(stderr, "no SM 6.6 device (%s); skipped\n", e.what());
        return;
    }
    const Host& host = *h;
    const fs::path dir = fs::temp_directory_path() / ("gpu-diff-cs-" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    std::vector<std::array<float, 4>> texels;
    for (int i = 0; i < 16; ++i) texels.push_back({float(i) / 16.0f, float(i % 4) * 0.3f, 0.5f, 1.0f});
    const std::array<float, 4> dummy[] = {{9.0f, 9.0f, 9.0f, 9.0f}};
    std::vector<std::array<float, 4>> cube_texels;
    for (int i = 0; i < 96; ++i)
        cube_texels.push_back({float(i / 16), float(i % 4) * 0.25f, float(i / 4 % 4) * 0.25f, 1.0f});
    // Buffers: 16 float4, 8 structures of 16 bytes, 32 raw dwords; the root CBV's 16 float4 and
    // the root SRV's 32 dwords.
    auto floats = [](std::size_t n, float scale) {
        std::vector<std::uint32_t> v;
        for (std::size_t i = 0; i < n; ++i) v.push_back(bits(float(i) * scale - 3.0f));
        return v;
    };
    const std::vector<std::uint32_t> typed_data = floats(64, 0.25f);
    std::vector<std::uint32_t> structured_data;
    for (std::uint32_t i = 0; i < 32; ++i)
        structured_data.push_back(i % 4 == 1 ? i * 0x0101'0101u : bits(float(i) * 0.5f));
    std::vector<std::uint32_t> raw_data;
    for (std::uint32_t i = 0; i < 32; ++i) raw_data.push_back(i * 0x9E37'79B9u);  // wrapping
    auto typed_view = [&](DXGI_FORMAT format, std::uint32_t bytes) {
        return View::buffer(raw_data, Stride::format_of(format, bytes));
    };
    const std::vector<View> views = {
        View::texture(1, 1, dummy),
        View::texture(4, 4, texels),
        View::buffer(typed_data, Stride::typed()),
        View::buffer(structured_data, Stride::structured(16)),
        View::buffer(raw_data, Stride::raw()),
        View::raw_uav(1024),
        View::texture_uav(16, 16),
        View::uint_texture_uav(16, 16),
        View::half_uav(256),
        typed_view(DXGI_FORMAT_R8G8B8A8_UNORM, 4),
        typed_view(DXGI_FORMAT_R16G16_SNORM, 4),
        typed_view(DXGI_FORMAT_R8G8B8A8_UINT, 4),
        typed_view(DXGI_FORMAT_R10G10B10A2_UNORM, 4),
        typed_view(DXGI_FORMAT_R16G16B16A16_UINT, 8),
        typed_view(DXGI_FORMAT_R8_UNORM, 1),
        typed_view(DXGI_FORMAT_R16_FLOAT, 2),
        typed_view(DXGI_FORMAT_R8G8B8A8_SNORM, 4),
        View::cube(4, 1, cube_texels),
    };
    const std::vector<std::uint32_t> root_cbv = floats(64, -0.125f);
    std::vector<std::uint32_t> root_srv;
    for (std::uint32_t i = 0; i < 32; ++i) root_srv.push_back(i * 0x0001'0003u + 7);
    D3D12_SAMPLER_DESC sd{};
    sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sd.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sd.MaxAnisotropy = 1;
    sd.MaxLOD = std::numeric_limits<float>::max();
    const D3D12_SAMPLER_DESC samplers[] = {sd, sd};
    auto dispatch = [&](std::vector<std::pair<std::uint32_t, Root>> roots) {
        Dispatch d;
        d.roots = std::move(roots);
        d.views = views;
        d.samplers = samplers;
        d.groups = {2, 2, 1};
        return d;
    };
    std::vector<std::string> failures;
    const std::optional<std::string> only = env("X2W_SHADERCFG_DIFF_ONLY");
    const bool warp = is_warp();
    for (const Case& c : CS_CASES) {
        const std::string_view cname = c.name;
        if (!selected(only, c.name)) continue;
        if (!(warp || !cname.starts_with("terminates"))) continue;
        const char* name = c.name;
        const std::string_view src = c.src;
        const std::vector<std::uint8_t> console_cs = compile(console, dir, src, "cs", "cs_6_5");
        if (const std::optional<std::string> out = env("X2W_SHADERCFG_EMIT_DUMP"))
            write_file(fs::path(*out) / (stem_of(name) + ".cs.cdxil"), console_cs);
        HostShader cs_c;
        try {
            cs_c = translate(console_cs);
        } catch (const ShaderError& e) {
            failures.push_back(std::string(name) + ": " + e.what());
            continue;
        }
        if (const std::optional<std::string> out = env("X2W_SHADERCFG_EMIT_DUMP"))
            write_file(fs::path(*out) / (stem_of(name) + ".cs.dxil"), cs_c.dxil);
        validate_if_there(cs_c.dxil, name);
        host::Pipeline pt;
        try {
            pt = host.compute_pipeline(cs_c);
        } catch (const host::Error& e) {
            // WARP runs no fixed wave size of 64.
            if (is_warp()) {
                std::fprintf(stderr, "%s: skipped on WARP (%s)\n", name, e.what());
                continue;
            }
            panic(std::string(name) + ": " + e.what());
        }
        const auto& [root_t, pso_t] = pt;
        // Our root views follow the user data in root parameter order.
        std::vector<std::pair<std::uint32_t, Root>> roots;
        roots.emplace_back(0, Root::constants(user_data(cs_c, K)));
        std::vector<std::pair<std::uint8_t, bool>> root_views;
        for (std::uint8_t s : cs_c.root_cbvs) root_views.emplace_back(s, false);
        for (std::uint8_t s : cs_c.root_srvs) root_views.emplace_back(s, true);
        std::sort(root_views.begin(), root_views.end());
        for (std::size_t n = 0; n < root_views.size(); ++n) {
            const auto [slot, srv] = root_views[n];
            if (slot != 4 + std::uint8_t(srv))
                panic(std::string(name) + ": root view: slot " + std::to_string(slot) + ", not " +
                      std::to_string(4 + int(srv)));
            roots.emplace_back(std::uint32_t(1 + n), Root::view(srv, srv ? root_srv : root_cbv));
        }
        std::optional<std::vector<std::vector<std::uint32_t>>> ours;
        std::string ours_error;
        try {
            ours = host.dispatch(root_t.Get(), pso_t.Get(), dispatch(std::move(roots)));
        } catch (const host::Error& e) {
            ours_error = e.what();
        }
        if (cname.starts_with("terminates")) {
            if (ours)
                std::fprintf(stderr, "%s: finished\n", name);
            else
                failures.push_back(std::string(name) + ": " + ours_error);
            continue;
        }
        if (!ours) panic(ours_error);
        const std::vector<std::uint8_t> cs_p = compile(pc, dir, src, "cs", "cs_6_5");
        const auto root_p = pc_root(host, cs_p);
        const auto pso_p = host.compute_pipeline_with(root_p.Get(), cs_p);
        std::vector<std::pair<std::uint32_t, Root>> roots_p;
        roots_p.emplace_back(0, Root::constants(std::vector<std::uint32_t>(K.begin(), K.end())));
        roots_p.emplace_back(1, Root::table(false, 1));
        roots_p.emplace_back(2, Root::table(true, 1));
        roots_p.emplace_back(3, Root::table(false, 5));
        roots_p.emplace_back(4, Root::view(false, root_cbv));
        roots_p.emplace_back(5, Root::view(true, root_srv));
        roots_p.emplace_back(6, Root::table(false, 9));
        const auto theirs = host.dispatch(root_p.Get(), pso_p.Get(), dispatch(std::move(roots_p)));
        // The first word that differs, over every UAV in order (Rust: flatten, zip, find).
        std::vector<std::uint32_t> a, b;
        for (const auto& u : *ours) a.insert(a.end(), u.begin(), u.end());
        for (const auto& u : theirs) b.insert(b.end(), u.begin(), u.end());
        std::optional<std::size_t> mismatch;
        for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
            if (!close_bits(a[i], b[i])) {
                mismatch = i;
                break;
            }
        if (!mismatch) {
            std::fprintf(stderr, "%s: equal\n", name);
        } else {
            const std::size_t at = *mismatch;
            failures.push_back(std::string(name) + ": word " + std::to_string(at) + ": " + hex(a[at]) + " (" +
                               display(std::bit_cast<float>(a[at])) + ") vs " + hex(b[at]) + " (" +
                               display(std::bit_cast<float>(b[at])) + ")");
        }
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    report(failures);
}

template <class F>
void run(const char* name, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        CHECK(false);
    }
}

}  // namespace

int main() {
    run("translated_shaders_draw_as_pc_ones", translated_shaders_draw_as_pc_ones);
    run("translated_mesh_shaders_draw_as_pc_ones", translated_mesh_shaders_draw_as_pc_ones);
    run("float_semantics_are_the_consoles", float_semantics_are_the_consoles);
    run("translated_compute_matches_pc", translated_compute_matches_pc);
    return test_util::finish("diff_test");
}
