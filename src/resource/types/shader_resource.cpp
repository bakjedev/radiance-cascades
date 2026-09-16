#include "shader_resource.hpp"
#include "src/file_system.hpp"

ShaderResource ShaderResourceLoader::operator()( const std::string& path ) const {
    auto data = file_system->read_binary(path);
    if (!data) {
        throw std::runtime_error("Failed to load shader: " + path);
    }
    return ShaderResource{.code = std::move(*data)};
}
