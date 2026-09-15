#include "file_system.hpp"

#include <fstream>

void FileSystem::add_route( std::string name, const std::filesystem::path& path ) {
    uri_.insert({std::move(name), path.lexically_normal()});
}

void FileSystem::remove_route( const std::string& name ) {
    if (auto it = uri_.find(name); it != uri_.end()) {
        uri_.erase(it);
    }
}

Result<std::filesystem::path> FileSystem::resolve( std::string_view virtual_path ) const {
    const std::filesystem::path raw{virtual_path};
    if (raw.is_absolute()) {
        return {raw.lexically_normal(), FileError::None};
    }

    if (auto colon = virtual_path.find(':'); colon != std::string_view::npos) {
        std::string route(virtual_path.substr(0, colon));
        if (auto it = uri_.find(route); it != uri_.end()) {
            std::string_view rest = virtual_path.substr(colon + 1);
            while (!rest.empty() && (rest.front() == '/' || rest.front() == '\\')) {
                rest.remove_prefix(1);
            }
            return {(it->second / rest).lexically_normal(), FileError::None};
        }
        if (route.size() > 1) {
            // could be C:
            return {{}, FileError::UnknownMount};
        }
    }

    if (!root_.empty()) {
        return {(root_ / raw).lexically_normal(), FileError::None};
    }

    return {raw.lexically_normal(), FileError::None};
}

bool FileSystem::exists( const std::string_view virtual_path ) const {
    auto result = resolve(virtual_path);
    if (!result) return false;
    std::error_code err_code;
    return std::filesystem::exists(*result, err_code) && !err_code;
}

bool FileSystem::is_file( const std::string_view virtual_path ) const {
    auto result = resolve(virtual_path);
    if (!result) return false;
    std::error_code err_code;
    return std::filesystem::is_regular_file(*result, err_code) && !err_code;
}

Result<std::string> FileSystem::read_text( const std::string_view virtual_path ) const {
    auto result = resolve(virtual_path);
    if (!result) return {{}, result.error};

    std::ifstream in;
    std::uintmax_t size = 0;

    if (auto err = open_for_read(*result, in, size); err != FileError::None) {
        return {{}, err};
    }

    std::string buf(size, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(size));

    if (in.bad() || static_cast<std::uintmax_t>(in.gcount()) != size)
        return {{}, FileError::ReadFailed};

    return {std::move(buf), FileError::None};
}

Result<std::vector<std::byte>> FileSystem::read_binary( const std::string_view virtual_path ) const {
    auto result = resolve(virtual_path);
    if (!result) return {{}, result.error};

    std::ifstream in;
    std::uintmax_t size = 0;

    if (auto err = open_for_read(*result, in, size); err != FileError::None) {
        return {{}, err};
    }

    std::vector<std::byte> buf(size);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));

    if (in.bad() || static_cast<std::uintmax_t>(in.gcount()) != size)
        return {{}, FileError::ReadFailed};

    return {std::move(buf), FileError::None};
}

FileError FileSystem::open_for_read( const std::filesystem::path& path, std::ifstream& out, std::uintmax_t& out_size ) {
    std::error_code err_code;

    if (!std::filesystem::exists(path, err_code)) {
        return FileError::NotFound;
    }

    if (!std::filesystem::is_regular_file(path, err_code)) {
        return FileError::NotAFile;
    }

    out_size = std::filesystem::file_size(path, err_code);

    if (err_code) {
        return FileError::ReadFailed;
    }

    out.open(path, std::ios::binary);

    if (!out) {
        return FileError::ReadFailed;
    }

    return FileError::None;
}
