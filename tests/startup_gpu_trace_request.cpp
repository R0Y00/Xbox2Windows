#include "startup_stubs/gpu_trace_request.h"

int main()
{
    using x2w_gpu::parse_scene_request;
    const auto plain = parse_scene_request(" \t+1\r\n");
    const auto keyed = parse_scene_request("2\t001abcDEF4321657\n");
    if (!plain || plain->id != 1 || !plain->stage_key.empty() ||
        !keyed || keyed->id != 2 || keyed->stage_key != "001abcDEF4321657") return 1;
    for (const auto input : {"", "0", "-1", "999999999999999999999", "2abcd", "2 0123456789ABCDEX",
                             "2 0000", "2 001abcDEF4321657 extra"})
        if (parse_scene_request(input)) return 2;
    if (parse_scene_request(std::string_view("1\0x", 3))) return 3;
    const auto pair = parse_scene_request("3 001abcDEF4321657\t f000000000000000\r\n");
    if (!pair || pair->stage_key != keyed->stage_key || pair->second_key != "f000000000000000") return 4;
    for (const auto input : {"4 001abcDEF4321657 001abcdef4321657", "4 f000000000000000 123",
                             "4 0000000000000001 0000000000000002 0000000000000003"})
        if (parse_scene_request(input)) return 5;
    const auto focused = parse_scene_request("5 001abcDEF4321657 f000000000000000 @300,450");
    if (!focused || !focused->point || (*focused->point)[0] != 300 || (*focused->point)[1] != 450 ||
        !parse_scene_request("1 @0,1000") || parse_scene_request("1 @-1,0") || parse_scene_request("1 @0,1001") ||
        parse_scene_request("1@1,1") || parse_scene_request("1 @2,3junk")) return 9;
    using x2w_gpu::stage_key_number;
    x2w_gpu::CaptureSelection selection{3, {stage_key_number(pair->stage_key), stage_key_number(pair->second_key)}, true};
    if (selection.matches(2, pair->second_key) || !selection.matches(3, "F000000000000000")) return 6;
    selection.complete(pair->second_key);
    if (selection.matches(3, pair->second_key) || !selection.matches(3, pair->stage_key)) return 7;
    selection.complete(pair->stage_key);
    return selection.id ? 8 : 0;
}
