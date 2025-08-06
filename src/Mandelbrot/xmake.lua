target("Mandelbrot")
    set_kind("binary")

    add_includedirs("$(env CONDA_INCLUDE)")
    add_includedirs("$(env NUMPY_CORE)/include")
    add_includedirs("$(projectdir)/dependencies/GLAD/include")
    add_includedirs("$(projectdir)/dependencies/matplotlib-cpp")

    add_files("*.cu")
    add_cugencodes("native")

    -- Anaconda
    add_links("python3")
    add_linkdirs("$(env PYTHON_LIB)")
target_end()