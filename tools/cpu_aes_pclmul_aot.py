"""Lift bounded Windows x64 leaf functions with legacy SSE & VEX.128 AES-NI and PCLMULQDQ to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned 16/32-byte buffers
(rcx and rdx are inputs, r8 is the output/in-out buffer). YMM0..YMM5 are modeled
as pure portable 256-bit value structs (four 64-bit lanes) with GF(2^8) AES tables/math
and GF(2) carryless multiplication (zero compiler intrinsics / zero <wmmintrin.h>).

Supported AES-NI and PCLMULQDQ instructions (docs/coverage/x64_zen2.tsv groups AES/Crypto):
  - Legacy SSE (preserves YMM_dst[255:128]):
      AESENC, AESENCLAST, AESDEC, AESDECLAST, AESIMC, AESKEYGENASSIST, PCLMULQDQ
  - VEX.128 (zeroes YMM_dst[255:128]):
      VAESENC, VAESENCLAST, VAESDEC, VAESDECLAST, VAESIMC, VAESKEYGENASSIST, VPCLMULQDQ

Transport / ABI glue:
  - MOVDQA / MOVDQU (legacy 128-bit reg/flat-[rcx|rdx|r8] loads, [r8] stores, reg-reg copies; preserves YMM[255:128])
  - VMOVDQA / VMOVDQU (VEX.128 128-bit zeroes YMM[255:128]; VEX.256 256-bit move glue only)
  - VZEROUPPER (zeroes YMM[255:128] of all defined XMMs)
  - terminal RET

VEX.256 compute ops (VAES*/VPCLMULQDQ on YMM), EVEX/XOP, nonvolatile XMM6-15/YMM6-15,
nonvolatile/unsupported GPRs, undefined XMM/YMM reads, displaced/indexed/RIP-relative/
non-argument memory operands, stores outside [r8], instruction prefixes, branches, and
missing or trailing instructions after ret are rejected.
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    FlowControl,
    MemorySizeExt,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(6, 32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)
NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.BX, R.BL, R.BH,
    R.RBP, R.EBP, R.BP, R.BPL,
    R.RSI, R.ESI, R.SI, R.SIL,
    R.RDI, R.EDI, R.DI, R.DIL,
    R.RSP, R.ESP, R.SP, R.SPL,
    R.R12, R.R12D, R.R12W, R.R12L,
    R.R13, R.R13D, R.R13W, R.R13L,
    R.R14, R.R14D, R.R14W, R.R14L,
    R.R15, R.R15D, R.R15W, R.R15L,
}

# (helper, arity, has_imm8, is_vex)
CRYPTO_SPECS = {
    Code.AESENC_XMM_XMMM128: ("op_aesenc", 2, False, False),
    Code.AESENCLAST_XMM_XMMM128: ("op_aesenclast", 2, False, False),
    Code.AESDEC_XMM_XMMM128: ("op_aesdec", 2, False, False),
    Code.AESDECLAST_XMM_XMMM128: ("op_aesdeclast", 2, False, False),
    Code.AESIMC_XMM_XMMM128: ("op_aesimc", 1, False, False),
    Code.AESKEYGENASSIST_XMM_XMMM128_IMM8: ("op_aeskeygenassist", 1, True, False),
    Code.PCLMULQDQ_XMM_XMMM128_IMM8: ("op_pclmulqdq", 2, True, False),
    Code.VEX_VAESENC_XMM_XMM_XMMM128: ("op_aesenc", 2, False, True),
    Code.VEX_VAESENCLAST_XMM_XMM_XMMM128: ("op_aesenclast", 2, False, True),
    Code.VEX_VAESDEC_XMM_XMM_XMMM128: ("op_aesdec", 2, False, True),
    Code.VEX_VAESDECLAST_XMM_XMM_XMMM128: ("op_aesdeclast", 2, False, True),
    Code.VEX_VAESIMC_XMM_XMMM128: ("op_aesimc", 1, False, True),
    Code.VEX_VAESKEYGENASSIST_XMM_XMMM128_IMM8: ("op_aeskeygenassist", 1, True, True),
    Code.VEX_VPCLMULQDQ_XMM_XMM_XMMM128_IMM8: ("op_pclmulqdq", 2, True, True),
}

# (is_ymm, is_vex)
MOVE_LOAD_CODES = {
    Code.MOVDQA_XMM_XMMM128: (False, False),
    Code.MOVDQU_XMM_XMMM128: (False, False),
    Code.VEX_VMOVDQA_XMM_XMMM128: (False, True),
    Code.VEX_VMOVDQU_XMM_XMMM128: (False, True),
    Code.VEX_VMOVDQA_YMM_YMMM256: (True, True),
    Code.VEX_VMOVDQU_YMM_YMMM256: (True, True),
}
MOVE_STORE_CODES = {
    Code.MOVDQA_XMMM128_XMM: (False, False),
    Code.MOVDQU_XMMM128_XMM: (False, False),
    Code.VEX_VMOVDQA_XMMM128_XMM: (False, True),
    Code.VEX_VMOVDQU_XMMM128_XMM: (False, True),
    Code.VEX_VMOVDQA_YMMM256_YMM: (True, True),
    Code.VEX_VMOVDQU_YMMM256_YMM: (True, True),
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    *CRYPTO_SPECS,
    *MOVE_LOAD_CODES,
    *MOVE_STORE_CODES,
}

PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_aes_pclmul_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 1) | (1 << 25) | (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
#endif

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

struct Vec256 {
    std::uint64_t q[4];
};

static inline Vec128 low128(const Vec256& v) {
    return {{v.q[0], v.q[1]}};
}

static inline void set_low128(Vec256& dst, const Vec128& v) {
    dst.q[0] = v.q[0];
    dst.q[1] = v.q[1];
}

static inline Vec256 zext256(const Vec128& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static inline Vec128 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    Vec128 out{};
    std::memcpy(out.q, reinterpret_cast<const void*>(address), sizeof(out.q));
    const volatile std::uint64_t q0 = out.q[0];
    const volatile std::uint64_t q1 = out.q[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    Vec256 out{};
    std::memcpy(out.q, reinterpret_cast<const void*>(address), sizeof(out.q));
    const volatile std::uint64_t q0 = out.q[0];
    const volatile std::uint64_t q1 = out.q[1];
    const volatile std::uint64_t q2 = out.q[2];
    const volatile std::uint64_t q3 = out.q[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline void store128(word address, const Vec128& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, sizeof(v.q));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store256(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, sizeof(v.q));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static constexpr std::uint8_t AES_SBOX[256] = {
    0x63u, 0x7Cu, 0x77u, 0x7Bu, 0xF2u, 0x6Bu, 0x6Fu, 0xC5u, 0x30u, 0x01u, 0x67u, 0x2Bu, 0xFEu, 0xD7u, 0xABu, 0x76u,
    0xCAu, 0x82u, 0xC9u, 0x7Du, 0xFAu, 0x59u, 0x47u, 0xF0u, 0xADu, 0xD4u, 0xA2u, 0xAFu, 0x9Cu, 0xA4u, 0x72u, 0xC0u,
    0xB7u, 0xFDu, 0x93u, 0x26u, 0x36u, 0x3Fu, 0xF7u, 0xCCu, 0x34u, 0xA5u, 0xE5u, 0xF1u, 0x71u, 0xD8u, 0x31u, 0x15u,
    0x04u, 0xC7u, 0x23u, 0xC3u, 0x18u, 0x96u, 0x05u, 0x9Au, 0x07u, 0x12u, 0x80u, 0xE2u, 0xEBu, 0x27u, 0xB2u, 0x75u,
    0x09u, 0x83u, 0x2Cu, 0x1Au, 0x1Bu, 0x6Eu, 0x5Au, 0xA0u, 0x52u, 0x3Bu, 0xD6u, 0xB3u, 0x29u, 0xE3u, 0x2Fu, 0x84u,
    0x53u, 0xD1u, 0x00u, 0xEDu, 0x20u, 0xFCu, 0xB1u, 0x5Bu, 0x6Au, 0xCBu, 0xBEu, 0x39u, 0x4Au, 0x4Cu, 0x58u, 0xCFu,
    0xD0u, 0xEFu, 0xAAu, 0xFBu, 0x43u, 0x4Du, 0x33u, 0x85u, 0x45u, 0xF9u, 0x02u, 0x7Fu, 0x50u, 0x3Cu, 0x9Fu, 0xA8u,
    0x51u, 0xA3u, 0x40u, 0x8Fu, 0x92u, 0x9Du, 0x38u, 0xF5u, 0xBCu, 0xB6u, 0xDAu, 0x21u, 0x10u, 0xFFu, 0xF3u, 0xD2u,
    0xCDu, 0x0Cu, 0x13u, 0xECu, 0x5Fu, 0x97u, 0x44u, 0x17u, 0xC4u, 0xA7u, 0x7Eu, 0x3Du, 0x64u, 0x5Du, 0x19u, 0x73u,
    0x60u, 0x81u, 0x4Fu, 0xDCu, 0x22u, 0x2Au, 0x90u, 0x88u, 0x46u, 0xEEu, 0xB8u, 0x14u, 0xDEu, 0x5Eu, 0x0Bu, 0xDBu,
    0xE0u, 0x32u, 0x3Au, 0x0Au, 0x49u, 0x06u, 0x24u, 0x5Cu, 0xC2u, 0xD3u, 0xACu, 0x62u, 0x91u, 0x95u, 0xE4u, 0x79u,
    0xE7u, 0xC8u, 0x37u, 0x6Du, 0x8Du, 0xD5u, 0x4Eu, 0xA9u, 0x6Cu, 0x56u, 0xF4u, 0xEAu, 0x65u, 0x7Au, 0xAEu, 0x08u,
    0xBAu, 0x78u, 0x25u, 0x2Eu, 0x1Cu, 0xA6u, 0xB4u, 0xC6u, 0xE8u, 0xDDu, 0x74u, 0x1Fu, 0x4Bu, 0xBDu, 0x8Bu, 0x8Au,
    0x70u, 0x3Eu, 0xB5u, 0x66u, 0x48u, 0x03u, 0xF6u, 0x0Eu, 0x61u, 0x35u, 0x57u, 0xB9u, 0x86u, 0xC1u, 0x1Du, 0x9Eu,
    0xE1u, 0xF8u, 0x98u, 0x11u, 0x69u, 0xD9u, 0x8Eu, 0x94u, 0x9Bu, 0x1Eu, 0x87u, 0xE9u, 0xCEu, 0x55u, 0x28u, 0xDFu,
    0x8Cu, 0xA1u, 0x89u, 0x0Du, 0xBFu, 0xE6u, 0x42u, 0x68u, 0x41u, 0x99u, 0x2Du, 0x0Fu, 0xB0u, 0x54u, 0xBBu, 0x16u,
};

static constexpr std::uint8_t AES_INV_SBOX[256] = {
    0x52u, 0x09u, 0x6Au, 0xD5u, 0x30u, 0x36u, 0xA5u, 0x38u, 0xBFu, 0x40u, 0xA3u, 0x9Eu, 0x81u, 0xF3u, 0xD7u, 0xFBu,
    0x7Cu, 0xE3u, 0x39u, 0x82u, 0x9Bu, 0x2Fu, 0xFFu, 0x87u, 0x34u, 0x8Eu, 0x43u, 0x44u, 0xC4u, 0xDEu, 0xE9u, 0xCBu,
    0x54u, 0x7Bu, 0x94u, 0x32u, 0xA6u, 0xC2u, 0x23u, 0x3Du, 0xEEu, 0x4Cu, 0x95u, 0x0Bu, 0x42u, 0xFAu, 0xC3u, 0x4Eu,
    0x08u, 0x2Eu, 0xA1u, 0x66u, 0x28u, 0xD9u, 0x24u, 0xB2u, 0x76u, 0x5Bu, 0xA2u, 0x49u, 0x6Du, 0x8Bu, 0xD1u, 0x25u,
    0x72u, 0xF8u, 0xF6u, 0x64u, 0x86u, 0x68u, 0x98u, 0x16u, 0xD4u, 0xA4u, 0x5Cu, 0xCCu, 0x5Du, 0x65u, 0xB6u, 0x92u,
    0x6Cu, 0x70u, 0x48u, 0x50u, 0xFDu, 0xEDu, 0xB9u, 0xDAu, 0x5Eu, 0x15u, 0x46u, 0x57u, 0xA7u, 0x8Du, 0x9Du, 0x84u,
    0x90u, 0xD8u, 0xABu, 0x00u, 0x8Cu, 0xBCu, 0xD3u, 0x0Au, 0xF7u, 0xE4u, 0x58u, 0x05u, 0xB8u, 0xB3u, 0x45u, 0x06u,
    0xD0u, 0x2Cu, 0x1Eu, 0x8Fu, 0xCAu, 0x3Fu, 0x0Fu, 0x02u, 0xC1u, 0xAFu, 0xBDu, 0x03u, 0x01u, 0x13u, 0x8Au, 0x6Bu,
    0x3Au, 0x91u, 0x11u, 0x41u, 0x4Fu, 0x67u, 0xDCu, 0xEAu, 0x97u, 0xF2u, 0xCFu, 0xCEu, 0xF0u, 0xB4u, 0xE6u, 0x73u,
    0x96u, 0xACu, 0x74u, 0x22u, 0xE7u, 0xADu, 0x35u, 0x85u, 0xE2u, 0xF9u, 0x37u, 0xE8u, 0x1Cu, 0x75u, 0xDFu, 0x6Eu,
    0x47u, 0xF1u, 0x1Au, 0x71u, 0x1Du, 0x29u, 0xC5u, 0x89u, 0x6Fu, 0xB7u, 0x62u, 0x0Eu, 0xAAu, 0x18u, 0xBEu, 0x1Bu,
    0xFCu, 0x56u, 0x3Eu, 0x4Bu, 0xC6u, 0xD2u, 0x79u, 0x20u, 0x9Au, 0xDBu, 0xC0u, 0xFEu, 0x78u, 0xCDu, 0x5Au, 0xF4u,
    0x1Fu, 0xDDu, 0xA8u, 0x33u, 0x88u, 0x07u, 0xC7u, 0x31u, 0xB1u, 0x12u, 0x10u, 0x59u, 0x27u, 0x80u, 0xECu, 0x5Fu,
    0x60u, 0x51u, 0x7Fu, 0xA9u, 0x19u, 0xB5u, 0x4Au, 0x0Du, 0x2Du, 0xE5u, 0x7Au, 0x9Fu, 0x93u, 0xC9u, 0x9Cu, 0xEFu,
    0xA0u, 0xE0u, 0x3Bu, 0x4Du, 0xAEu, 0x2Au, 0xF5u, 0xB0u, 0xC8u, 0xEBu, 0xBBu, 0x3Cu, 0x83u, 0x53u, 0x99u, 0x61u,
    0x17u, 0x2Bu, 0x04u, 0x7Eu, 0xBAu, 0x77u, 0xD6u, 0x26u, 0xE1u, 0x69u, 0x14u, 0x63u, 0x55u, 0x21u, 0x0Cu, 0x7Du,
};

static inline std::uint8_t aes_gf_mul(std::uint8_t a, std::uint8_t b) {
    std::uint8_t p = 0u;
    for (int i = 0; i < 8; ++i) {
        if ((b & 1u) != 0u) {
            p ^= a;
        }
        const bool hi = (a & 0x80u) != 0u;
        a = static_cast<std::uint8_t>(a << 1);
        if (hi) {
            a ^= 0x1Bu;
        }
        b >>= 1;
    }
    return p;
}

static inline void unpack_bytes(const Vec128& v, std::uint8_t out[16]) {
    std::memcpy(out, v.q, 16);
}

static inline Vec128 pack_bytes(const std::uint8_t in[16]) {
    Vec128 v{};
    std::memcpy(v.q, in, 16);
    return v;
}

static inline Vec128 xor128(const Vec128& a, const Vec128& b) {
    return {{a.q[0] ^ b.q[0], a.q[1] ^ b.q[1]}};
}

static inline void aes_shift_sub(const std::uint8_t in[16], std::uint8_t out[16]) {
    static constexpr unsigned idx[16] = {0u, 5u, 10u, 15u, 4u, 9u, 14u, 3u, 8u, 13u, 2u, 7u, 12u, 1u, 6u, 11u};
    for (unsigned i = 0u; i < 16u; ++i) {
        out[i] = AES_SBOX[in[idx[i]]];
    }
}

static inline void aes_inv_shift_sub(const std::uint8_t in[16], std::uint8_t out[16]) {
    static constexpr unsigned idx[16] = {0u, 13u, 10u, 7u, 4u, 1u, 14u, 11u, 8u, 5u, 2u, 15u, 12u, 9u, 6u, 3u};
    for (unsigned i = 0u; i < 16u; ++i) {
        out[i] = AES_INV_SBOX[in[idx[i]]];
    }
}

static inline void aes_mix_columns(const std::uint8_t in[16], std::uint8_t out[16]) {
    for (unsigned c = 0u; c < 4u; ++c) {
        const std::uint8_t a0 = in[c * 4u + 0u];
        const std::uint8_t a1 = in[c * 4u + 1u];
        const std::uint8_t a2 = in[c * 4u + 2u];
        const std::uint8_t a3 = in[c * 4u + 3u];
        out[c * 4u + 0u] = static_cast<std::uint8_t>(aes_gf_mul(2u, a0) ^ aes_gf_mul(3u, a1) ^ a2 ^ a3);
        out[c * 4u + 1u] = static_cast<std::uint8_t>(a0 ^ aes_gf_mul(2u, a1) ^ aes_gf_mul(3u, a2) ^ a3);
        out[c * 4u + 2u] = static_cast<std::uint8_t>(a0 ^ a1 ^ aes_gf_mul(2u, a2) ^ aes_gf_mul(3u, a3));
        out[c * 4u + 3u] = static_cast<std::uint8_t>(aes_gf_mul(3u, a0) ^ a1 ^ a2 ^ aes_gf_mul(2u, a3));
    }
}

static inline void aes_inv_mix_columns(const std::uint8_t in[16], std::uint8_t out[16]) {
    for (unsigned c = 0u; c < 4u; ++c) {
        const std::uint8_t a0 = in[c * 4u + 0u];
        const std::uint8_t a1 = in[c * 4u + 1u];
        const std::uint8_t a2 = in[c * 4u + 2u];
        const std::uint8_t a3 = in[c * 4u + 3u];
        out[c * 4u + 0u] = static_cast<std::uint8_t>(aes_gf_mul(0x0Eu, a0) ^ aes_gf_mul(0x0Bu, a1) ^ aes_gf_mul(0x0Du, a2) ^ aes_gf_mul(0x09u, a3));
        out[c * 4u + 1u] = static_cast<std::uint8_t>(aes_gf_mul(0x09u, a0) ^ aes_gf_mul(0x0Eu, a1) ^ aes_gf_mul(0x0Bu, a2) ^ aes_gf_mul(0x0Du, a3));
        out[c * 4u + 2u] = static_cast<std::uint8_t>(aes_gf_mul(0x0Du, a0) ^ aes_gf_mul(0x09u, a1) ^ aes_gf_mul(0x0Eu, a2) ^ aes_gf_mul(0x0Bu, a3));
        out[c * 4u + 3u] = static_cast<std::uint8_t>(aes_gf_mul(0x0Bu, a0) ^ aes_gf_mul(0x0Du, a1) ^ aes_gf_mul(0x09u, a2) ^ aes_gf_mul(0x0Eu, a3));
    }
}

static inline Vec128 op_aesenc(const Vec128& state, const Vec128& key) {
    std::uint8_t b[16]{}, tmp[16]{}, mixed[16]{};
    unpack_bytes(state, b);
    aes_shift_sub(b, tmp);
    aes_mix_columns(tmp, mixed);
    return xor128(pack_bytes(mixed), key);
}

static inline Vec128 op_aesenclast(const Vec128& state, const Vec128& key) {
    std::uint8_t b[16]{}, tmp[16]{};
    unpack_bytes(state, b);
    aes_shift_sub(b, tmp);
    return xor128(pack_bytes(tmp), key);
}

static inline Vec128 op_aesdec(const Vec128& state, const Vec128& key) {
    std::uint8_t b[16]{}, tmp[16]{}, mixed[16]{};
    unpack_bytes(state, b);
    aes_inv_shift_sub(b, tmp);
    aes_inv_mix_columns(tmp, mixed);
    return xor128(pack_bytes(mixed), key);
}

static inline Vec128 op_aesdeclast(const Vec128& state, const Vec128& key) {
    std::uint8_t b[16]{}, tmp[16]{};
    unpack_bytes(state, b);
    aes_inv_shift_sub(b, tmp);
    return xor128(pack_bytes(tmp), key);
}

static inline Vec128 op_aesimc(const Vec128& src) {
    std::uint8_t b[16]{}, mixed[16]{};
    unpack_bytes(src, b);
    aes_inv_mix_columns(b, mixed);
    return pack_bytes(mixed);
}

static inline std::uint32_t aes_sub_word(std::uint32_t w) {
    return static_cast<std::uint32_t>(AES_SBOX[w & 0xFFu])
         | (static_cast<std::uint32_t>(AES_SBOX[(w >> 8) & 0xFFu]) << 8)
         | (static_cast<std::uint32_t>(AES_SBOX[(w >> 16) & 0xFFu]) << 16)
         | (static_cast<std::uint32_t>(AES_SBOX[(w >> 24) & 0xFFu]) << 24);
}

static inline Vec128 op_aeskeygenassist(const Vec128& src, std::uint8_t imm8) {
    std::uint32_t w[4]{};
    std::memcpy(w, src.q, 16);
    const std::uint32_t s1 = aes_sub_word(w[1]);
    const std::uint32_t s3 = aes_sub_word(w[3]);
    const std::uint32_t rcon = static_cast<std::uint32_t>(imm8);
    const std::uint32_t out[4] = {
        s1,
        ((s1 >> 8) | (s1 << 24)) ^ rcon,
        s3,
        ((s3 >> 8) | (s3 << 24)) ^ rcon,
    };
    Vec128 res{};
    std::memcpy(res.q, out, 16);
    return res;
}

static inline Vec128 op_pclmulqdq(const Vec128& src1, const Vec128& src2, std::uint8_t imm8) {
    const std::uint64_t a = ((imm8 & 0x01u) != 0u) ? src1.q[1] : src1.q[0];
    const std::uint64_t b = ((imm8 & 0x10u) != 0u) ? src2.q[1] : src2.q[0];
    std::uint64_t lo = 0ull;
    std::uint64_t hi = 0ull;
    for (unsigned i = 0u; i < 64u; ++i) {
        if (((b >> i) & 1ull) != 0ull) {
            lo ^= (a << i);
            if (i != 0u) {
                hi ^= (a >> (64u - i));
            }
        }
    }
    return {{lo, hi}};
}
"""


def vec_index(register, ins):
    """Map volatile XMM0..XMM5/YMM0..YMM5 to (index, is_ymm), rejecting nonvolatile/unsupported."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, expected_bytes, *, store=False):
    """Lower a flat zero-displacement [rcx|rdx|r8] memory operand (stores only to [r8])."""
    if ins.is_ip_rel_memory_operand or ins.memory_base in (R.RIP, R.EIP):
        raise ValueError(f"RIP-relative memory operand at RVA {ins.ip:#x}: {ins}")
    if ins.memory_index != R.NONE:
        raise ValueError(f"SIB index memory operand at RVA {ins.ip:#x}: {ins}")
    base = ins.memory_base
    if base == R.NONE or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS:
        if base in NONVOLATILE_GPR:
            raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
        raise ValueError(f"bad base register in memory operand at RVA {ins.ip:#x}: {ins}")
    if ins.memory_displacement != 0:
        raise ValueError(f"displaced memory operand at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    actual_bytes = MemorySizeExt.size(ins.memory_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"unexpected {actual_bytes}-byte memory operand (expected {expected_bytes}) at RVA {ins.ip:#x}: {ins}"
        )
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AES-NI / PCLMULQDQ leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_xmm = set()
    defined_ymm_hi = set()
    statements = []
    returned = False

    def read_vec(op_index, ins, *, require_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != require_ymm:
            raise ValueError(f"unexpected vector register width {ins.op_register(op_index)} at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_xmm or (require_ymm and idx not in defined_ymm_hi):
            reg_label = f"ymm{idx}" if require_ymm else f"xmm{idx}"
            raise ValueError(f"undefined vector register {reg_label} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}" if require_ymm else f"low128(ymm{idx})"

    def read_rm128(op_index, ins):
        kind = ins.op_kind(op_index)
        if kind == OpKind.REGISTER:
            return read_vec(op_index, ins, require_ymm=False)
        if kind == OpKind.MEMORY:
            addr = flat_address(ins, 16, store=False)
            load_var = f"ld_{ins.ip:x}_{op_index}"
            statements.append(f"const Vec128 {load_var} = load128({addr});")
            return load_var
        raise ValueError(f"unsupported operand at RVA {ins.ip:#x}: {ins}")

    def read_rm256(op_index, ins):
        kind = ins.op_kind(op_index)
        if kind == OpKind.REGISTER:
            return read_vec(op_index, ins, require_ymm=True)
        if kind == OpKind.MEMORY:
            addr = flat_address(ins, 32, store=False)
            load_var = f"ld_{ins.ip:x}_{op_index}"
            statements.append(f"const Vec256 {load_var} = load256({addr});")
            return load_var
        raise ValueError(f"unsupported operand at RVA {ins.ip:#x}: {ins}")

    def check_dst_vec(op_index, ins, *, expect_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != expect_ymm:
            raise ValueError(f"unexpected vector destination width at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if (
            ins.has_lock_prefix
            or ins.has_rep_prefix
            or ins.has_repne_prefix
            or ins.has_segment_prefix
            or ins.has_xacquire_prefix
            or ins.has_xrelease_prefix
        ):
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")

        op_code = ins.op_code()
        encoding = op_code.encoding
        if encoding == EncodingKind.EVEX or ins.op_mask != R.NONE or ins.zeroing_masking or ins.is_broadcast:
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if encoding == EncodingKind.XOP:
            raise ValueError(f"unsupported XOP instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code != Code.RETNQ and ins.flow_control != FlowControl.NEXT:
            raise ValueError(f"non-leaf branch or call at RVA {ins.ip:#x}: {ins}")
        if encoding == EncodingKind.VEX and op_code.l == 1 and ins.code not in (
            Code.VEX_VMOVDQA_YMM_YMMM256,
            Code.VEX_VMOVDQA_YMMM256_YMM,
            Code.VEX_VMOVDQU_YMM_YMMM256,
            Code.VEX_VMOVDQU_YMMM256_YMM,
        ):
            raise ValueError(f"unsupported VEX.256 instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        code = ins.code
        if code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_xmm):
                statements.append(f"ymm{idx}.q[2] = 0ull;")
                statements.append(f"ymm{idx}.q[3] = 0ull;")
                defined_ymm_hi.add(idx)
        elif code in CRYPTO_SPECS:
            helper, arity, has_imm8, is_vex = CRYPTO_SPECS[code]
            expected_enc = EncodingKind.VEX if is_vex else EncodingKind.LEGACY
            if encoding != expected_enc:
                raise ValueError(f"unexpected encoding for {helper} at RVA {ins.ip:#x}: {ins}")
            if is_vex:
                if arity == 2:
                    arg1 = read_vec(1, ins, require_ymm=False)
                    arg2 = read_rm128(2, ins)
                    args = [arg1, arg2]
                else:
                    args = [read_rm128(1, ins)]
            else:
                if arity == 2:
                    arg1 = read_vec(0, ins, require_ymm=False)
                    arg2 = read_rm128(1, ins)
                    args = [arg1, arg2]
                else:
                    args = [read_rm128(1, ins)]
            if has_imm8:
                imm8 = ins.immediate8 & 0xFF
                args.append(f"{imm8:#x}u")
            dst_idx = check_dst_vec(0, ins, expect_ymm=False)
            call_expr = f"{helper}({', '.join(args)})"
            if is_vex:
                statements.append(f"ymm{dst_idx} = zext256({call_expr});")
                defined_ymm_hi.add(dst_idx)
            else:
                statements.append(f"set_low128(ymm{dst_idx}, {call_expr});")
            defined_xmm.add(dst_idx)
        elif code in MOVE_LOAD_CODES:
            is_ymm, is_vex = MOVE_LOAD_CODES[code]
            if is_ymm:
                src_expr = read_rm256(1, ins)
                dst_idx = check_dst_vec(0, ins, expect_ymm=True)
                statements.append(f"ymm{dst_idx} = {src_expr};")
                defined_ymm_hi.add(dst_idx)
            else:
                src_expr = read_rm128(1, ins)
                dst_idx = check_dst_vec(0, ins, expect_ymm=False)
                if is_vex:
                    statements.append(f"ymm{dst_idx} = zext256({src_expr});")
                    defined_ymm_hi.add(dst_idx)
                else:
                    statements.append(f"set_low128(ymm{dst_idx}, {src_expr});")
            defined_xmm.add(dst_idx)
        elif code in MOVE_STORE_CODES:
            is_ymm, is_vex = MOVE_STORE_CODES[code]
            src_expr = read_vec(1, ins, require_ymm=is_ymm)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, 32 if is_ymm else 16, store=True)
                fn = "store256" if is_ymm else "store128"
                statements.append(f"{fn}({addr}, {src_expr});")
            else:
                dst_idx = check_dst_vec(0, ins, expect_ymm=is_ymm)
                if is_ymm:
                    statements.append(f"ymm{dst_idx} = {src_expr};")
                    defined_ymm_hi.add(dst_idx)
                elif is_vex:
                    statements.append(f"ymm{dst_idx} = zext256({src_expr});")
                    defined_ymm_hi.add(dst_idx)
                else:
                    statements.append(f"set_low128(ymm{dst_idx}, {src_expr});")
                defined_xmm.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


def _same_path(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a.resolve(strict=False) == b.resolve(strict=False)


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise ValueError(f"input image is not a regular file: {args.image}")
        names = [n for n in args.exports.split(",") if n]
        if not names:
            raise ValueError("expected at least one export name")
        if args.output.is_symlink():
            raise ValueError(f"output path cannot be a symlink: {args.output}")
        if args.output.exists() and args.output.is_dir():
            raise ValueError(f"output path is a directory: {args.output}")
        if _same_path(args.image, args.output):
            raise ValueError(f"output path matches input image: {args.output}")
        source, meta = translate(args.image, names)
        _write_atomic(args.output, source)
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
        return 0
    except (ValueError, OSError, pefile.PEFormatError) as error:
        sys.stderr.write(f"{error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
