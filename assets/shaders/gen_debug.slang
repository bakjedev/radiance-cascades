static const float PI = 3.14159265359;

struct PushConstant {
    uint vertex_id;
    uint cascade_id;
    uint cascade;
    uint cascade_width;
    uint cascade_height;
    uint base_probe_size;
    float base_spacing;
    uint base_probe_dir_count;
    float base_length;
}

[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];

struct DebugLineVertex {
    float3 color;
    float2 pos;
};

[[vk::binding(1, 0)]] RWStructuredBuffer<DebugLineVertex> storageBuffers[];

static const float3 colors[] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
    float3(0.0, 1.0, 1.0)
};

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.cascade_width, pc.cascade_height))) return;

    int probe_size = int(pc.base_probe_size) << pc.cascade;
    int2 probe = local / probe_size;
    float2 origin = (float2(probe) + 0.5) * (pc.base_spacing * float(1 << pc.cascade));

    int2 interval = local % probe_size;
    int dir_index = interval.y * probe_size + interval.x;
    float angle = 2.0 * PI * ((float(dir_index) + 0.5) / float(pc.base_probe_dir_count << (2 * pc.cascade)));
    float2 dir = float2(cos(angle), sin(angle));

    float start_t = pc.base_length * ((pc.cascade == 0) ? 0.0 : float(1 << (2 * pc.cascade)));
    float end_t = pc.base_length * float(1 << (2 * (pc.cascade + 1)));

    float2 from = origin + dir * start_t;
    float2 to = origin + dir * end_t;

    float4 hit_col = storageImageArraysF[pc.cascade_id].Load(int3(local, pc.cascade));

    float3 color = colors[pc.cascade];
    if (any(hit_col.xyz > float3(0.0))) {
        color = float3(1.0);
    }

    int offset = int(pc.cascade_width * pc.cascade_height * pc.cascade);
    int index = local.y * int(pc.cascade_width) + local.x + offset;

    storageBuffers[pc.vertex_id][2 * index].pos = from;
    storageBuffers[pc.vertex_id][2 * index].color = color;

    storageBuffers[pc.vertex_id][2 * index + 1].pos = to;
    storageBuffers[pc.vertex_id][2 * index + 1].color = color;
}