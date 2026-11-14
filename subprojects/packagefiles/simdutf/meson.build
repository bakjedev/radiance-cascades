project('simdutf', 'cpp', version : '9.3.3', default_options : ['cpp_std=c++20', 'warning_level=0', 'werror=false'])

inc = include_directories('include')
simdutf_lib = static_library('simdutf', 'src/simdutf.cpp',
                             include_directories : [inc, include_directories('src')],
)

simdutf_dep = declare_dependency(link_with : simdutf_lib, include_directories : inc)
meson.override_dependency('simdutf', simdutf_dep)