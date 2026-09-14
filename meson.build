project('radiance-cascades', 'cpp', version : '0.1', meson_version : '>=1.1.0',
        default_options : [
            'cpp_std=c++20',
            'warning_level=3',
            'werror=true',
            'buildtype=debug'
        ]
)

compiler = meson.get_compiler('cpp')
add_project_arguments(compiler.get_supported_arguments(
                          '-Wshadow', '-Wnon-virtual-dtor', '-Wold-style-cast', '-Wcast-align',
                          '-Wunused', '-Woverloaded-virtual', '-Wconversion', '-Wsign-conversion',
                          '-Wnull-dereference', '-Wdouble-promotion', '-Wformat=2',
                          '-Wimplicit-fallthrough', '-Wmisleading-indentation', '-Wduplicated-cond',
                          '-Wduplicated-branches', '-Wlogical-op', '-Wuseless-cast', '-Wsuggest-override'
                      ), language : 'cpp')

framework_dep = dependency('framework')

vulkan_dep = dependency('vulkan', include_type : 'system')
sdl3_dep = dependency('sdl3')
vma_dep = dependency('VulkanMemoryAllocator')

sources = files(
    'src/main.cpp',
    'src/application.cpp',
    'src/window.cpp',
    'src/input.cpp',
    'src/render2d/renderer_2d.cpp',
)

executable('radiance-cascades', sources, dependencies : [framework_dep, vulkan_dep, sdl3_dep, vma_dep])
