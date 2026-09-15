#pragma once

struct ShaderResource {
    std::vector<std::byte> code;
};


struct ShaderResourceLoader {
    ShaderResource operator()( const std::string& path ) const {
        auto data = file_system->read_binary(path);
        if (!data) {
            throw std::runtime_error("Failed to load shader: " + path);
        }
        return ShaderResource{.code = std::move(*data)};
    }

    FileSystem* file_system;
};
