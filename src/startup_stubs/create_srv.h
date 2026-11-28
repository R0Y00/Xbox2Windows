// The desktop SRV boundary: every shader resource view the stand-in writes goes through x2w_create_srv, which repairs
// fields D3D12 would reject (view_desc_check.h) or writes a null view of the same shape instead of an invalid one.
#pragma once

#include "startup_stubs/view_desc_check.h"

#include <windows.h>

#include <cstdio>
#include <cstring>

namespace x2w_startup {

using SrvLog = void (*)(const char*, ...);

inline void srv_diagnostic(const char* site, const char* action, const char* field, UINT from, UINT to, SrvLog log)
{
    static LONG lines;
    if (InterlockedIncrement(&lines) > 64) return;
    char line[256];
    std::snprintf(line, sizeof(line), "x2w-d3d12 SRV %s: site=%s field/reason=%s %u -> %u", action, site, field, from, to);
    if (log) {
        log("%s", line);
    } else {
        OutputDebugStringA(line);
        OutputDebugStringA("\n");
    }
}

inline D3D12_SHADER_RESOURCE_VIEW_DESC safe_null_srv(D3D12_SRV_DIMENSION dimension, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = dimension;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    switch (dimension) {
    case D3D12_SRV_DIMENSION_BUFFER: view.Buffer.NumElements = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1D: view.Texture1D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: view.Texture1DArray = {0, 1, 0, 1, 0}; break;
    case D3D12_SRV_DIMENSION_TEXTURE2D: view.Texture2D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: view.Texture2DArray = {0, 1, 0, 1, 0, 0}; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS: break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY: view.Texture2DMSArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: view.Texture3D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: view.TextureCube.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: view.TextureCubeArray = {0, 1, 0, 1, 0}; break;
    case D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE: view.Format = DXGI_FORMAT_UNKNOWN; break;
    default:
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        break;
    }
    return view;
}

inline bool texture_srv_dimension(D3D12_SRV_DIMENSION dimension)
{
    return dimension >= D3D12_SRV_DIMENSION_TEXTURE1D && dimension <= D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
}

// The device can sample or load `format` through a view of `dimension`.
inline bool srv_format_supported(ID3D12Device* device, DXGI_FORMAT format, D3D12_SRV_DIMENSION dimension)
{
    if (format == DXGI_FORMAT_UNKNOWN || is_typeless(format) || view_desc::depth_format(format) ||
        format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010 || format == DXGI_FORMAT_P016 ||
        format == DXGI_FORMAT_420_OPAQUE)
        return false;
    D3D12_FORMAT_SUPPORT1 required;
    switch (dimension) {
    case D3D12_SRV_DIMENSION_BUFFER: required = D3D12_FORMAT_SUPPORT1_BUFFER; break;
    case D3D12_SRV_DIMENSION_TEXTURE1D:
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: required = D3D12_FORMAT_SUPPORT1_TEXTURE1D; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: required = D3D12_FORMAT_SUPPORT1_TEXTURE3D; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: required = D3D12_FORMAT_SUPPORT1_TEXTURECUBE; break;
    default: required = D3D12_FORMAT_SUPPORT1_TEXTURE2D; break;
    }
    D3D12_FEATURE_DATA_FORMAT_SUPPORT caps{format};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &caps, sizeof(caps))) &&
           (caps.Support1 & required) &&
           (caps.Support1 & (D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE));
}

// Checks (and repairs in place) a view of a real resource; nullptr = the resource's default view.
inline view_desc::Refusal validate_srv(ID3D12Device* device, D3D12_SHADER_RESOURCE_VIEW_DESC* view,
                                      const D3D12_RESOURCE_DESC& resource, const char* site, SrvLog log)
{
    using view_desc::Refusal;
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
    if (view && view->ViewDimension != D3D12_SRV_DIMENSION_BUFFER && view->Format != resource.Format &&
        !is_typeless(resource.Format) && compatible_formats(resource.Format, view->Format))
        device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options, sizeof(options));
    const auto refusal = view ? view_desc::check(*view, resource,
        [&](view_desc::Field field, UINT from, UINT to) {
            srv_diagnostic(site, "repaired", view_desc::field_name(field), from, to, log);
        }, options.CastingFullyTypedFormatSupported != FALSE) : view_desc::check_default(resource);
    if (refusal != Refusal::none) return refusal;
    const auto dimension = view ? view->ViewDimension : view_desc::default_dimension(resource);
    const auto format = view ? view->Format : resource.Format;
    if (dimension == D3D12_SRV_DIMENSION_BUFFER && view &&
        (view->Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW || view->Buffer.StructureByteStride))
        return Refusal::none;
    return srv_format_supported(device, format, dimension) ? Refusal::none : Refusal::format;
}

// A null view (no resource): only its shape matters. Repairs `checked` to canonical null coordinates.
inline view_desc::Refusal check_null_srv(ID3D12Device* device, D3D12_SHADER_RESOURCE_VIEW_DESC& checked, const char* site,
                                        SrvLog log)
{
    using view_desc::Refusal;
    const auto dimension = checked.ViewDimension;
    if (dimension == D3D12_SRV_DIMENSION_UNKNOWN) return Refusal::dimension;
    if (dimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) {
        if (!view_desc::valid_mapping(checked.Shader4ComponentMapping)) {
            srv_diagnostic(site, "repaired", view_desc::field_name(view_desc::Field::mapping),
                           checked.Shader4ComponentMapping, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, log);
            checked.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
        return checked.Format != DXGI_FORMAT_UNKNOWN ||
               checked.RaytracingAccelerationStructure.Location % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT
            ? Refusal::format : Refusal::none;
    }
    if (dimension == D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_RESOURCE_DESC bounds{};
        bounds.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bounds.Width = ~UINT64{0};
        const auto refusal = validate_srv(device, &checked, bounds, site, log);
        if (refusal == Refusal::none) {
            checked.Buffer.FirstElement = 0;
            checked.Buffer.NumElements = 1;
        }
        return refusal;
    }
    if (!texture_srv_dimension(dimension)) return Refusal::dimension;
    // Null textures have no subresources: keep the shader-visible type and use canonical coordinates.
    if (!srv_format_supported(device, checked.Format, dimension)) return Refusal::format;
    auto canonical = safe_null_srv(dimension, checked.Format);
    if (view_desc::valid_mapping(checked.Shader4ComponentMapping))
        canonical.Shader4ComponentMapping = checked.Shader4ComponentMapping;
    else
        srv_diagnostic(site, "repaired", view_desc::field_name(view_desc::Field::mapping),
                       checked.Shader4ComponentMapping, canonical.Shader4ComponentMapping, log);
    checked = canonical;
    return Refusal::none;
}

// The null view written instead of a refused one: the same dimension (and buffer kind) the binding expects where it is
// known, the resource's default dimension otherwise, in a format the device supports.
inline D3D12_SHADER_RESOURCE_VIEW_DESC refused_srv_replacement(ID3D12Device* device, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                                              const D3D12_RESOURCE_DESC& backing, bool has_resource)
{
    auto dimension = desc ? desc->ViewDimension : view_desc::default_dimension(backing);
    auto format = desc ? desc->Format : backing.Format;
    if (dimension != D3D12_SRV_DIMENSION_BUFFER && !texture_srv_dimension(dimension) &&
        !(dimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE && !has_resource))
        dimension = view_desc::default_dimension(backing);
    if (dimension == D3D12_SRV_DIMENSION_UNKNOWN) dimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    if (dimension != D3D12_SRV_DIMENSION_BUFFER) {
        if (!srv_format_supported(device, format, dimension)) format = DXGI_FORMAT_R32_FLOAT;
        return safe_null_srv(dimension, format);
    }
    // Keep the binding's raw or structured kind when it can be identified.
    auto view = safe_null_srv(dimension, format);
    if (desc && desc->ViewDimension == dimension) {
        view.Buffer.Flags = desc->Buffer.Flags;
        view.Buffer.StructureByteStride = desc->Buffer.StructureByteStride;
    }
    if (view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW) {
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.Buffer.StructureByteStride = 0;
    } else if (!buffer_srv_element_bytes(view) ||
               (!view.Buffer.StructureByteStride && !srv_format_supported(device, format, dimension))) {
        view = safe_null_srv(dimension, DXGI_FORMAT_R32_UINT);
    }
    return view;
}

// True if the requested view was written (possibly repaired). A refusal always overwrites the destination with a null
// view, so the slot never keeps an older view the caller meant to replace.
inline bool x2w_create_srv(ID3D12Device* device, ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                           D3D12_CPU_DESCRIPTOR_HANDLE dest, const char* site, SrvLog log = nullptr)
{
    using view_desc::Refusal;
    const D3D12_RESOURCE_DESC backing = resource ? resource->GetDesc() : D3D12_RESOURCE_DESC{};
    D3D12_SHADER_RESOURCE_VIEW_DESC checked{};
    if (desc) std::memcpy(&checked, desc, sizeof(checked));
    Refusal refusal = Refusal::dimension;
    if (resource) refusal = validate_srv(device, desc ? &checked : nullptr, backing, site, log);
    else if (desc) refusal = check_null_srv(device, checked, site, log);
    if (refusal != Refusal::none) {
        srv_diagnostic(site, "refused (writing null)", view_desc::refusal_field_name(refusal), 0, 0, log);
        checked = refused_srv_replacement(device, desc, backing, resource != nullptr);
        device->CreateShaderResourceView(nullptr, &checked, dest);
        return false;
    }
    device->CreateShaderResourceView(resource, desc ? &checked : nullptr, dest);
    return true;
}

}  // namespace x2w_startup
