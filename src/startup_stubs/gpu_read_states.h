#pragma once
#include <d3d12.h>

namespace x2w_gpu {
inline D3D12_RESOURCE_STATES texture_read_state(D3D12_COMMAND_LIST_TYPE type)
{
    return type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                    : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
}

inline D3D12_RESOURCE_STATES buffer_read_state(D3D12_COMMAND_LIST_TYPE type)
{
    auto state = texture_read_state(type) | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
    return state;
}
} // namespace x2w_gpu
