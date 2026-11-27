// The draw stand-in accepts a numrecords udsrc and refuses an unknown kind or a gap.
// Exit codes are the cases tests/startup_gfx2hlsl_ps_raw_descriptor_check.py reports.
#include "udsrc_line.h"

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
    return 0;
}
