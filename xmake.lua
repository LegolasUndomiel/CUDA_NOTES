-- 工程名称
set_project("CUDA_NOTES")

-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编译标准
set_languages("c17","cxx17")

set_targetdir("bin/$(mode)")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")
-- 自动更新compile_commands.json
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(builddir)")
    os.rm(target:targetdir())
end)

add_includedirs("$(env CUDA_PATH)/include")

includes("**/xmake.lua")
