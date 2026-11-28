#include "common.slang"

struct PushConstant {
    uint indirect_id;
    uint geometries_id;
    uint primitive_id;
    uint count_id;
    uint filter_id;
    uint geometries_count;
}

struct Primitive {
    uint first_index;
    uint index_count;
    uint vertex_offset;
    uint pad;
    float4 local_bounds;
}

struct Command {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

struct Filter {
    float4 planes[6];
    uint include_mask;
    uint exclude_mask;
    uint cull_flags;
    uint base;
}

namespace CullFlags {
    static const uint None = 0;
    static const uint Frustum = 1U << 0;
    static const uint Near = 1U << 1;
}

[[vk::binding(4, 0)]] ConstantBuffer<FrameData> frameDataUBOs[];
[[vk::binding(1, 0)]] RWStructuredBuffer<Command> storageBuffersCommand[];
[[vk::binding(1, 0)]] StructuredBuffer<Geometry> storageBuffersGeometry[];
[[vk::binding(1, 0)]] StructuredBuffer<Primitive> storageBuffersPrimitive[];
[[vk::binding(1, 0)]] StructuredBuffer<Filter> storageBuffersFilter[];
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> storageBuffersCount[];

[shader("compute")]
[numthreads(256)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    uint index = dispatchThreadID.x;
    if (index >= pc.geometries_count) return;

    Geometry geometry = storageBuffersGeometry[pc.geometries_id][index];
    Primitive primitive = storageBuffersPrimitive[pc.primitive_id][geometry.primitive];

    uint filter_index = dispatchThreadID.y;
    Filter filter = storageBuffersFilter[pc.filter_id][filter_index];

    if ((geometry.flags & filter.include_mask) != filter.include_mask) return;
    if ((geometry.flags & filter.exclude_mask) != 0) return;

    if ((filter.cull_flags & CullFlags::Frustum) != 0) {
        uint count = (filter.cull_flags & CullFlags::Near) != 0 ? 6 : 5;
        for (uint p = 0; p < count; p++)
            if (dot(filter.planes[p].xyz, geometry.bounds.xyz) + filter.planes[p].w < -geometry.bounds.w) return;
    }

    uint slot;
    InterlockedAdd(storageBuffersCount[pc.count_id][filter_index], 1, slot);

    storageBuffersCommand[pc.indirect_id][filter.base + slot].index_count = primitive.index_count;
    storageBuffersCommand[pc.indirect_id][filter.base + slot].instance_count = 1;
    storageBuffersCommand[pc.indirect_id][filter.base + slot].first_index = primitive.first_index;
    storageBuffersCommand[pc.indirect_id][filter.base + slot].vertex_offset = primitive.vertex_offset;
    storageBuffersCommand[pc.indirect_id][filter.base + slot].first_instance = index;
}