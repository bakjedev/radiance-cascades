project(
    'ImGui Plotting Library',
    'cpp',
    license : 'MIT',
    version : '1.0',
    meson_version : '>=0.56.0',
)

imgui_sp = subproject('imgui')
imgui_dep = imgui_sp.get_variable('imgui_dep')

implot_inc = include_directories('.')
implot_src = files('implot.cpp', 'implot_items.cpp')

cpp = meson.get_compiler('cpp')
implot_args = cpp.get_supported_arguments(
    '-Wno-deprecated-enum-enum-conversion',
    '/wd5054',
)

implot_lib = library(
    'implot',
    implot_src,
    cpp_args : implot_args,
    dependencies : imgui_dep,
)

implot_dep = declare_dependency(
    link_with : implot_lib,
    include_directories : implot_inc,
)

meson.override_dependency('implot', implot_dep)
