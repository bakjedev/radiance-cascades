#pragma once

struct ShaderResource {
    std::vector<uint32_t> code;
};


struct ShaderResourceLoader {
    ShaderResource operator()( const std::string& ) const {
        return {};
    }
};
