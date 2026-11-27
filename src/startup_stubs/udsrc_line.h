#pragma once
// Manifest `udsrc` lines for a translated draw. Kind numbers are Translation::UdKind in d3d12_draw.inl.
#include <cstdio>
#include <cstring>

enum {
    udsrc_rect = 0,
    udsrc_base_vertex = 1,
    udsrc_start_instance = 2,
    udsrc_root_constant = 3,
    udsrc_stride = 4,
    udsrc_mrt_bits = 5,
    udsrc_d16_unorm16 = 6,
    udsrc_desc_word3 = 7,
    udsrc_num_records = 8
};

struct UdSrcParsed {
    bool is_udsrc = false;
    int kind = -1;
    int index = 0;
    int a = 0;
    int b = 0;
};

// A line that is not `udsrc` leaves is_udsrc clear and returns true so the caller can try another kind of line.
// An unknown kind, a gap, or an index that is not the next slot returns false: the translation is not used,
// so a later udsrc line is not dropped while the shader still runs.
// mrtbits is known only when the manifest is paired with an RTV request. d16unorm16 needs a view index in 0..31.
inline bool accept_udsrc_line(const char* line, int next_index, bool rtv_request, UdSrcParsed* out)
{
    out->is_udsrc = false;
    out->kind = -1;
    if (std::strncmp(line, "udsrc ", 6) != 0) return true;
    int index = 0, a = 0, b = 0;
    char source[16] = "";
    if (std::sscanf(line, "udsrc %d %15s %d %d", &index, source, &a, &b) < 2) return false;
    int kind = -1;
    if (!std::strcmp(source, "rect")) kind = udsrc_rect;
    else if (!std::strcmp(source, "basevertex")) kind = udsrc_base_vertex;
    else if (!std::strcmp(source, "startinstance")) kind = udsrc_start_instance;
    else if (!std::strcmp(source, "rootconst")) kind = udsrc_root_constant;
    else if (!std::strcmp(source, "stride")) kind = udsrc_stride;
    else if (!std::strcmp(source, "d16unorm16") && a >= 0 && a < 32) kind = udsrc_d16_unorm16;
    else if (!std::strcmp(source, "descword3")) kind = udsrc_desc_word3;
    else if (!std::strcmp(source, "mrtbits") && rtv_request) kind = udsrc_mrt_bits;
    else if (!std::strcmp(source, "numrecords")) kind = udsrc_num_records;
    if (kind < 0 || index != next_index || next_index < 0 || next_index >= 32) return false;
    out->is_udsrc = true;
    out->kind = kind;
    out->index = index;
    out->a = a;
    out->b = b;
    return true;
}
