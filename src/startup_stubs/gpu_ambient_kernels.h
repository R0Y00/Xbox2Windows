// GPU kernels for screen-space indirect illumination.
#pragma once

namespace x2w_gpu::ambient::detail {
inline constexpr char source[] = R"hlsl(
cbuffer LightingSettings : register(b0) {
    float intensity, fixedLight, creaseStrength, creaseRadius;
    float nearPlane, verticalSlope; uint2 extent;
    float adaptation, maximumRatio, fogRatio, reserved;
};
Texture2D<float> sceneDepth : register(t0);
RWStructuredBuffer<float4> history : register(u1);
RWStructuredBuffer<float4> coefficients : register(u2);
static const uint2 cellGrid = uint2(64,36);
static const uint2 probeGrid = uint2(160,90);
static const uint cellCount = 2304;
static const float3 lumaWeights = float3(0.2126,0.7152,0.0722);

float3 eyePosition(int2 pixel) {
    float sampleDepth = sceneDepth.Load(int3(clamp(pixel, 0, int2(extent)-1), 0));
    float distance = min(1000.0, nearPlane / max(sampleDepth, 1e-7));
    float2 uv = (pixel + 0.5) / float2(extent);
    float2 ray = (2.0 * uv - 1.0) * float2(float(extent.x)/extent.y, -1) * verticalSlope;
    return float3(ray,1) * distance;
}
float3 gridPosition(uint2 bin, uint2 grid) {
    return eyePosition(int2((bin + 0.5) * float2(extent) / grid));
}

#ifdef MEASURE
RWTexture2D<float3> sceneColor : register(u0);
[numthreads(8,4,1)]
void Measure(uint3 tid : SV_DispatchThreadID) {
    uint2 resolution;
    sceneColor.GetDimensions(resolution.x, resolution.y);
    float3 energy = 0;
    for (uint j = 0; j != 25; ++j) {
        float2 offset = (float2(j % 5, j / 5) + 0.5) / 5.0;
        uint2 samplePixel = uint2((tid.xy + offset) * resolution / cellGrid);
        float3 value = max(sceneColor[samplePixel], 0);
        if (any(isinf(value)) || any(isnan(value))) value = 0;
        energy += value / max(1.0, dot(value,lumaWeights) / 20.0);
    }
    history[tid.y * cellGrid.x + tid.x] = float4(energy / 25, 1);
}
[numthreads(1,1,1)]
void Adapt() {
    float3 average = 0;
    for (uint i = 0; i < cellCount; ++i) average += history[i].rgb;
    average /= cellCount;
    float4 old = history[cellCount];
    history[cellCount] = float4(old.a > 0 ? lerp(old.rgb, average, adaptation) : average, 1);
}
#endif

#ifdef PROBES
[numthreads(8,4,1)]
void Locate(uint3 tid : SV_DispatchThreadID) {
    float3 pos = gridPosition(tid.xy, cellGrid);
    history[cellCount + 1 + tid.x + cellGrid.x * tid.y] = float4(pos, pos.z * pos.z);
}
[numthreads(8,2,1)]
void Integrate(uint3 tid : SV_DispatchThreadID) {
    float3 pos = gridPosition(tid.xy, probeGrid);
    float softening = 0.25 + 0.0025 * pos.z * pos.z;
    float sumWeight = 0;
    float3 sumDirection = 0, sumEnergy = 0;
    float3 momentR = 0, momentG = 0, momentB = 0;
    for (uint i = 0; i < cellCount; ++i) {
        float4 cell = history[cellCount + 1 + i];
        float3 difference = cell.xyz - pos;
        float dist2 = dot(difference,difference);
        float weight = cell.w / (dist2 + softening);
        float3 direction = difference / sqrt(dist2 + 0.25 * softening);
        float3 light = history[i].rgb;
        sumWeight += weight;
        sumDirection += weight * direction;
        sumEnergy += weight * light;
        momentR += weight * light.r * direction;
        momentG += weight * light.g * direction;
        momentB += weight * light.b * direction;
    }
    sumWeight = max(sumWeight, 1e-6);
    float3 energy = sumEnergy / sumWeight;
    float3 meanDirection = sumDirection / sumWeight;
    uint start = 3 * (tid.y * probeGrid.x + tid.x);
    coefficients[start]   = float4(momentR / sumWeight - energy.r * meanDirection, energy.r);
    coefficients[start+1] = float4(momentG / sumWeight - energy.g * meanDirection, energy.g);
    coefficients[start+2] = float4(momentB / sumWeight - energy.b * meanDirection, energy.b);
}
#endif

#ifdef SCATTER
RWTexture3D<float3> fog : register(u0);
[numthreads(8,8,2)]
void LimitFog(uint3 voxel : SV_DispatchThreadID) {
    uint3 size;
    fog.GetDimensions(size.x,size.y,size.z);
    if (any(voxel >= size)) return;
    uint2 probe = min(uint2((voxel.xy + 0.5) * probeGrid / float2(size.xy)),probeGrid-1);
    uint start = 3 * (probe.x + probe.y * probeGrid.x);
    float3 localAverage = float3(coefficients[start].w,coefficients[start+1].w,coefficients[start+2].w);
    float cap = fogRatio * lerp(dot(history[cellCount].rgb,lumaWeights),dot(localAverage,lumaWeights),
                                float(voxel.z) / size.z);
    float3 value = fog[voxel];
    float brightness = dot(value,lumaWeights);
    if (brightness > cap) fog[voxel] = value * (cap/brightness);
}
#endif

#ifdef RESOLVE
RWTexture2D<uint> indirectLight : register(u0);
float3 decodeRGB9E5(uint bits) {
    return float3(bits & 511, (bits >> 9) & 511, (bits >> 18) & 511) * exp2(float(bits >> 27)-24);
}
uint encodeRGB9E5(float3 value) {
    value = clamp(value, 0, 65408);
    float peak = max(value.r,max(value.g,value.b));
    int exponent = clamp((int)floor(log2(max(peak,1e-9))) + 16,0,31);
    if (peak * exp2(float(24 - exponent)) >= 511.5) exponent = min(31,exponent+1);
    uint3 m = min(uint3(value * exp2(float(24 - exponent)) + 0.5),511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(exponent) << 27);
}
float3 slope(int2 p, int2 step, float3 center) {
    float3 back = center - eyePosition(p-step);
    float3 forward = eyePosition(p+step) - center;
    return abs(back.z) < abs(forward.z) ? back : forward;
}
float openness(int2 pixel, float3 center, float3 normal, float noise) {
    float footprint = min(128.0, creaseRadius * extent.y / (2 * verticalSlope * center.z));
    if (creaseStrength <= 0 || footprint < 1.5 || center.z >= 400) return 1;
    float sum = 0;
    for (int j = 0; j < 12; ++j) {
        float angle = j * 2.399963 + noise * 6.2831853;
        float2 offset = float2(cos(angle),sin(angle)) * (sqrt((j + 0.5)/12) * footprint);
        float3 delta = eyePosition(pixel + int2(round(offset))) - center;
        float d2 = dot(delta,delta);
        float above = max(0.0, dot(delta,normal) - 0.004 * center.z - 0.005);
        sum += above / sqrt(d2 + 1e-6) * saturate(1.0 - d2/(creaseRadius*creaseRadius));
    }
    return saturate(1.0 - creaseStrength * 2 * sum / 12);
}
[numthreads(8,8,1)]
void ResolveDiffuse(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= extent)) return;
    int2 pixel = int2(thread.xy);
    float3 center = eyePosition(pixel);
    float3 normal = normalize(cross(slope(pixel,int2(0,1),center),slope(pixel,int2(1,0),center)) + float3(0,0,-1e-9));
    if (dot(normal,center) > 0) normal = -normal;
    float jitter = frac(52.9829189 * frac(dot(float2(pixel),float2(0.06711056,0.00583715))));
    float visibility = openness(pixel,center,normal,jitter);
    if (fixedLight > 0) {
        indirectLight[pixel] = encodeRGB9E5(float3(1.0,0.98,0.94) * fixedLight * visibility);
        return;
    }
    float2 at = (pixel + 0.5) * probeGrid / float2(extent) - 0.5 + 0.5 * (jitter-0.5);
    int2 base = int2(floor(at));
    float2 blend = at-base;
    float weights = 0;
    float4 r = 0, g = 0, b = 0;
    for (int i = 0; i < 4; ++i) {
        int2 corner = int2(i&1,i>>1);
        uint2 probe = uint2(clamp(base+corner,0,int2(probeGrid)-1));
        float dz = abs(gridPosition(probe,probeGrid).z-center.z)/center.z;
        float2 bilinear = lerp(1.0-blend,blend,float2(corner));
        float w = (bilinear.x * bilinear.y + 0.001) / (0.02 + dz);
        uint idx = 3 * (probe.x + probe.y * probeGrid.x);
        r += w * coefficients[idx];
        g += w * coefficients[idx+1];
        b += w * coefficients[idx+2];
        weights += w;
    }
    float3 diffuse = float3(r.w,g.w,b.w) / weights;
    float3 directional = diffuse + 1.2 * float3(dot(r.xyz,normal),dot(g.xyz,normal),dot(b.xyz,normal)) / weights;
    float3 estimate = lerp(clamp(directional,0.2*diffuse,2.5*diffuse),history[cellCount].rgb,0.2)
                    * intensity * visibility;
    float3 prior = decodeRGB9E5(indirectLight[pixel]);
    if (prior.g < 0.5 * min(prior.r,prior.b) && max(prior.r,prior.b) > 0.02) prior = 0;
    float ceiling = max(dot(estimate,lumaWeights) * maximumRatio,1e-6);
    prior *= min(1.0,ceiling / max(dot(prior,lumaWeights),1e-6));
    indirectLight[pixel] = encodeRGB9E5(max(prior,estimate));
}
#endif
)hlsl";
}
