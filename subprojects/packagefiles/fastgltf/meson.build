project('fastgltf', 'cpp',
        version : '1.0.1',
        meson_version : '>=1.1',
        default_options : ['cpp_std=c++20', 'warning_level=0', 'werror=false'],
)

simdjson_dep = dependency('simdjson')
simdutf_dep = dependency('simdutf')

inc = include_directories('include')

fastgltf_lib = static_library('fastgltf',
                              'src/base64.cpp',
                              'src/crc32.cpp',
                              'src/exporter.cpp',
                              'src/io.cpp',
                              'src/parser.cpp',
                              'src/uri.cpp',
                              'src/validate.cpp',
                              include_directories : inc,
                              dependencies : [simdjson_dep, simdutf_dep],
)

fastgltf_dep = declare_dependency(
    link_with : fastgltf_lib,
    include_directories : inc,
)
meson.override_dependency('fastgltf', fastgltf_dep)