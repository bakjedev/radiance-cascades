project('stb', 'c')

stb_lib = static_library('stb',
                         'stb_impl.c',
                         include_directories : include_directories('.'),
)

stb_dep = declare_dependency(
    include_directories : include_directories('.', is_system : true),
    link_with : stb_lib,
)