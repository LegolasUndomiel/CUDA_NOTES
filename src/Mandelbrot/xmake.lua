target("mandelbrotStatic")
    set_kind("static")
    -- 编译CUDA静态库时必须这样设置
    -- 启用设备链接以支持静态库设备代码链接
    set_policy("build.cuda.devlink", true)

    add_files("mandelbrot.cu")

    -- CUDA
    add_cugencodes("native")
    if is_plat("linux") then
        -- 为静态库添加-fPIC选项，因为它被用于构建动态库
        add_cuflags("-Xcompiler -fPIC")
        add_culdflags("-Xcompiler -fPIC")
        -- OpenMP
        add_cuflags("-Xcompiler -fopenmp")
        add_culdflags("-Xcompiler -fopenmp")
    end
    if is_plat("windows") then
        set_runtimes("MD")
        -- OpenMP
        add_cuflags("-Xcompiler /openmp")
        add_culdflags("-Xcompiler /openmp")
    end
target_end()

if is_plat("linux") then
    -- OpenMP
    add_ldflags("-fopenmp")
end
if is_plat("windows") then
    set_runtimes("MD")
    -- OpenMP
    add_ldflags("/openmp")
end

target("mandelbrotBinary")
    set_kind("binary")
    add_files("main.cu")

    add_deps("mandelbrotStatic")
target_end()

target("mandelbrot")
    -- 这里编译出来的动态库的文件名需要和Python模块名相同
    -- 否则Python导入模块时会报错
    -- 当target的名称和Python模块名不同时，可以通过set_basename来设置
    set_basename("mandelbrot")

    set_kind("shared")
    add_rules("python.module")

    add_includedirs("$(env CONDA_INCLUDE)")
    add_includedirs("$(projectdir)/dependencies/pybind11/include")

    add_files("pybind.cc")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env PYTHON_LIB)")

    add_deps("mandelbrotStatic")
target_end()