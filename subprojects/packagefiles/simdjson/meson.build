project('simdjson', 'cpp', version : '5.1.0', default_options : ['cpp_std=c++20', 'warning_level=0', 'werror=false'])

inc = include_directories('singleheader')
simdjson_lib = static_library('simdjson', 'singleheader/simdjson.cpp', include_directories : inc)

simdjson_dep = declare_dependency(link_with : simdjson_lib, include_directories : inc)
meson.override_dependency('simdjson', simdjson_dep)