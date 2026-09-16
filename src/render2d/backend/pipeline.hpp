#pragma once
#include <vulkan/vulkan.hpp>

inline vk::UniquePipelineLayout create_pipeline_layout(
    vk::Device device,
    std::span<const vk::DescriptorSetLayout> sets = {},
    std::span<const vk::PushConstantRange> push = {} ) {
    return device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{}
        .setSetLayouts(sets)
        .setPushConstantRanges(push)
    );
}

struct ShaderStage {
    vk::ShaderStageFlagBits stage;
    vk::ShaderModule module;
    std::string entry = "main";
    const vk::SpecializationInfo* specialization = nullptr;
};

struct ComputePipelineDesc {
    ShaderStage shader_stage;
    vk::PipelineLayout layout;
    vk::PipelineCreateFlags flags{};
};

inline vk::UniquePipeline create_compute_pipeline( vk::Device device, const ComputePipelineDesc& desc,
                                                   vk::UniquePipelineCache cache = {} ) {
    vk::PipelineShaderStageCreateInfo stage{};
    stage.setStage(vk::ShaderStageFlagBits::eCompute);
    stage.setModule(desc.shader_stage.module);
    stage.setPName(desc.shader_stage.entry.c_str());
    stage.setPSpecializationInfo(desc.shader_stage.specialization);

    vk::ComputePipelineCreateInfo create_info{};
    create_info.setFlags(desc.flags);
    create_info.setStage(stage);
    create_info.setLayout(desc.layout);

    auto [result, pipeline] = device.createComputePipelineUnique(cache.get(), create_info);
    if (result != vk::Result::eSuccess) {
        return {};
    }
    return std::move(pipeline);
}

inline constexpr auto blend_opaque = [] {
    vk::PipelineColorBlendAttachmentState blend{};
    blend.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA);
    return blend;
}();


inline constexpr auto blend_alpha = [] {
    vk::PipelineColorBlendAttachmentState blend{};
    blend.setBlendEnable(vk::True);
    blend.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha);
    blend.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha);
    blend.setColorBlendOp(vk::BlendOp::eAdd);
    blend.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
    blend.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
    blend.setAlphaBlendOp(vk::BlendOp::eAdd);
    return blend;
}();

struct GraphicsPipelineDesc {
    std::vector<ShaderStage> stages;

    std::vector<vk::VertexInputBindingDescription> vertex_bindings;
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes;

    vk::PipelineInputAssemblyStateCreateInfo input_assembly{{}, vk::PrimitiveTopology::eTriangleList};

    vk::PipelineRasterizationStateCreateInfo rasterization{};

    vk::PipelineMultisampleStateCreateInfo multisample{};

    vk::PipelineDepthStencilStateCreateInfo depth_stencil{{}, vk::True, vk::True, vk::CompareOp::eLess};

    std::vector<vk::Format> color_formats;
    std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments;
    std::vector<vk::DynamicState> dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::Format depth_format = vk::Format::eUndefined;
    vk::Format stencil_format = vk::Format::eUndefined;

    vk::PipelineLayout layout;
    vk::PipelineCreateFlags flags{};

    GraphicsPipelineDesc& add_attachment( const vk::Format format,
                                          const vk::PipelineColorBlendAttachmentState& blend_state = blend_opaque ) {
        color_formats.push_back(format);
        blend_attachments.push_back(blend_state);
        return *this;
    }
};

inline vk::UniquePipeline create_graphics_pipeline( vk::Device device, const GraphicsPipelineDesc& desc,
                                                    vk::UniquePipelineCache cache = {} ) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.reserve(desc.stages.size());
    for (auto& stage : desc.stages) {
        stages.emplace_back(vk::PipelineShaderStageCreateFlags{}, stage.stage, stage.module, stage.entry.c_str(),
                            stage.specialization);
    }

    vk::PipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.setVertexBindingDescriptions(desc.vertex_bindings);
    vertex_input.setVertexAttributeDescriptions(desc.vertex_attributes);

    vk::PipelineViewportStateCreateInfo viewport{};
    viewport.setScissorCount(1);
    viewport.setViewportCount(1);

    vk::PipelineColorBlendStateCreateInfo color_blend{};
    color_blend.setAttachments(desc.blend_attachments);

    vk::PipelineDynamicStateCreateInfo dynamic{};
    dynamic.setDynamicStates(desc.dynamic_states);

    vk::PipelineRenderingCreateInfo rendering{};
    rendering.setColorAttachmentFormats(desc.color_formats);
    rendering.setDepthAttachmentFormat(desc.depth_format);

    vk::GraphicsPipelineCreateInfo create_info{};
    create_info.setFlags(desc.flags);
    create_info.setStages(stages);
    create_info.setPVertexInputState(&vertex_input);
    create_info.setPInputAssemblyState(&desc.input_assembly);
    create_info.setPViewportState(&viewport);
    create_info.setPRasterizationState(&desc.rasterization);
    create_info.setPMultisampleState(&desc.multisample);
    create_info.setPDepthStencilState(&desc.depth_stencil);
    create_info.setPColorBlendState(&color_blend);
    create_info.setPDynamicState(&dynamic);
    create_info.setLayout(desc.layout);
    create_info.setPNext(&rendering);

    auto [result, pipeline] = device.createGraphicsPipelineUnique(cache.get(), create_info);
    if (result != vk::Result::eSuccess) {
        return {};
    }
    return std::move(pipeline);
}
