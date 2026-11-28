// The draw stand-in accepts a numrecords udsrc and refuses an unknown kind or a gap.
// Exit codes are the cases tests/startup_gfx2hlsl_ps_raw_descriptor_check.py reports.
#include "udsrc_line.h"
#include "indirect_buffer_view.h"

struct HostResource {};

int main()
{
    UdSrcParsed parsed{};
    if (!accept_udsrc_line("udsrc 0 rect", 0, false, &parsed) || !parsed.is_udsrc || parsed.kind != udsrc_rect)
        return 1;
    if (!accept_udsrc_line("udsrc 4 numrecords 0", 4, false, &parsed) || parsed.kind != udsrc_num_records || parsed.a != 0)
        return 2;
    if (accept_udsrc_line("udsrc 5 notakind 0", 5, false, &parsed))
        return 3;
    if (accept_udsrc_line("udsrc 6 stride 0", 5, false, &parsed))
        return 4;
    if (!accept_udsrc_line("udsrc 5 stride 0", 5, false, &parsed) || parsed.kind != udsrc_stride)
        return 5;
    if (!accept_udsrc_line("srv 0 buf table 0 0", 0, false, &parsed) || parsed.is_udsrc)
        return 6;
    if (accept_udsrc_line("udsrc 0 mrtbits", 0, false, &parsed))
        return 7;
    if (!accept_udsrc_line("udsrc 0 mrtbits", 0, true, &parsed) || parsed.kind != udsrc_mrt_bits)
        return 8;
    HostResource resource;
    const auto null_input = indirect_buffer_view(&resource, std::uint64_t{8}, std::uint64_t{16}, 0u, 0u, true, 3);
    if (null_input.records != 0 || !null_input.upload || null_input.source != 3 || null_input.bytes != 16 || null_input.offset != 8)
        return 9;
    const auto snapshot = indirect_buffer_view(&resource, std::uint64_t{4}, std::uint64_t{32}, 8u, 4u, true, 2);
    if (snapshot.records != 4 || !snapshot.upload || snapshot.source != 2 || snapshot.stride != 8)
        return 10;
    const auto copied = indirect_buffer_view(&resource, std::uint64_t{0}, std::uint64_t{32}, 8u, 7u, false, 1);
    if (copied.records != 7 || copied.upload || copied.source != 1 || copied.resource != &resource)
        return 11;
    return 0;
}
