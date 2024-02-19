local projectsDir = {
    "PluginInteropUnityCUDA/",
    "SampleBasic/",
    "Utilities/",
}

-- Define the "clean" action
newaction {
    trigger     = "clean",
    description = "Clean the project",
    execute     = function()
        cwd = os.getcwd()
        local pathCleanPs1 = ".//buildtools//clean.ps1"
        print("Run " .. pathCleanPs1 .. " from " .. cwd)
        -- Create a list of all projects to pass as arguments
        local projectDirList = table.concat(projectsDir, ",")

        -- Call the PowerShell script with required arguments
        os.execute('powershell -NoProfile -ExecutionPolicy Bypass -File ' ..
            pathCleanPs1 .. ' -projectDirs "' .. projectDirList .. '" -solutionDir "' .. cwd .. '"')
    end
}

function loadrequire(module, linkToRepository)
    local function requiref(module)
        require(module)
    end

    res = pcall(requiref, module)
    if not (res) then
        print("Could not find module :"
        , module, " it's available at this link : "
        , linkToRepository)
    end
end

function addCUDAToProject(sourceDir, objDir)
    buildcustomizations "BuildCustomizations/CUDA 12.2"
    -- CUDA specific properties

    cudaIntDir(objDir)
    cudaFiles
    {
        sourceDir .. "/**.cu"
    }

    cudaFastMath "On"
    cudaRelocatableCode "On"
    cudaMaxRegCount "32"

    -- Let's compile for all supported architectures (and also in parallel with -t0)
    cudaCompilerOptions { "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86", "-t0", "-Wreorder", "-Werror=reorder", "-err-no", "-Xcompiler=/wd4505" }

    filter "configurations:debug"
    cudaLinkerOptions { "-g", "-Xcompiler=/wd4100" }
    filter {}

    filter "configurations:release"
    cudaFastMath "On"
    filter {}
end

local root = "./"
local pathTargetDebug = "bin/Debug/"
local pathTargetRelease = "bin/Release/"
local pathSampleProject = "SampleBasic"
local pathUtilities = root .. "Utilities"
local pathPluginInterop = root .. "PluginInteropUnityCUDA"
local nameUtilitiesLib = "Utilities.lib"
local nameUtilitiesLibDebug = "d_Utilities.lib"
local namePluginInteropLib = "PluginInteropUnityCUDA.lib"
local namePluginInteropLibDebug = "d_PluginInteropUnityCUDA.lib"
local pathUtilitiesInclude = pathUtilities .. "/include/"
local pathPluginInteropInclude = pathPluginInterop .. "/include/"
local pathPluginInteropIncludeSubdir = pathPluginInteropInclude .. "**"
local pathThirdPartyUnity = pathPluginInterop .. "/thirdParty/unity/include/"
local pathThirdPartyGLDir = pathPluginInterop .. "/thirdParty/gl3w/include/";
local pathSourceThirdPartyGLDir = pathPluginInterop .. "/thirdParty/gl3w/src/";

loadrequire('premake-cuda\\premake5-cuda', "https://github.com/theComputeKid/premake5-cuda")
print(
    "We use export compile commands module to export compilation database for clang. If you don't have the module you won't be able to compile with clang, BUT you can still compile with visual studio !")
loadrequire('export-compile-commands\\export-compile-commands',
    "https://github.com/null-black/premake-export-compile-commands")
---------------------------------
-- [ WORKSPACE CONFIGURATION   --
---------------------------------
-- Setting up the workspace. A workspace can have multiple Projects.
-- In visual studio a workspace corresponds to a solution.
workspace "PluginInteropUnity"

configurations { "Debug", "Release" } -- Optimization/General config mode in VS
platforms { "x64" }                   -- Dropdown platforms section in VS
-------------------------------
-- [ COMPILER/LINKER CONFIG] --
-------------------------------
-- Here we are setting up what differentiates the configurations that we called "Debug" and "Release"
filter "configurations:Debug"
defines { "DEBUG" }
symbols "On"
filter "configurations:Release"
defines { "NDEBUG" }
optimize "On"
filter { "platforms:*64" }
architecture "x64"

defines { "PLUGINGLINTEROPCUDA_SHARED_EXPORTS" }
symbols "On"

-- building makefiles
filter { "action:gmake" }

-- building make files on mac specifically
filter { "system:macosx", "action:gmake" }
toolset "clang"
-- Toolset changes the compiler/linker/etc of the current project configuration.
--     "clang" is used in this case, because premake defaults to using gcc/g++
--     when used with the 'gmake' action, but macosx ships with clang by default
--     and if we don't set it explicitly we will get some warnings.

filter {} -- clear filter when you know you no longer need it!

-------------------------------
-- [ PluginGLInteropCUDA ] --
-------------------------------
project "PluginInteropUnityCUDA"
local rootProject = pathPluginInterop
location(rootProject)
language "C++"

filter "configurations:Debug"
targetdir(pathTargetDebug, "SampleBasic")
targetname("d_PluginInteropUnityCUDA")
filter "configurations:Release"
targetdir(pathTargetRelease, "SampleBasic")
filter {}

local tempDir = rootProject .. "/temp/"
objdir(tempDir)
kind "SharedLib"

local SourceDir = rootProject .. "/src/";
local IncludeSubDir = pathPluginInteropIncludeSubdir;
local IncludeDir = pathPluginInteropInclude;

dependson { "Utilities" }
files
{
    SourceDir .. "**.c",
    pathSourceThirdPartyGLDir .. "**.c",
    pathThirdPartyGLDir .. "**.h",
    SourceDir .. "**.cpp",
    IncludeDir .. "**.h",
    IncludeDir .. "**.hpp",
    IncludeSubDir .. "**.h",
    IncludeSubDir .. "**.hpp"
}

includedirs
{
    IncludeDir,
    IncludeSubDir,
    pathThirdPartyGLDir,
    pathThirdPartyUnity,
    pathUtilitiesInclude,
}

filter "configurations:Debug"

libdirs
{
    pathTargetDebug
}
links { nameUtilitiesLibDebug, "OpenGL32" }

filter "configurations:Release"

libdirs
{
    pathTargetRelease
}
links { nameUtilitiesLib, "OpenGL32" }

filter {}




sourceDirAbsolute = path.getabsolute(SourceDir)
objDirAbsolute = path.getabsolute(tempDir)
addCUDAToProject(sourceDirAbsolute, objDirAbsolute)



-------------------------------
-- [ SampleBasic ] --
-------------------------------
project "SampleBasic"
local locationProject = pathSampleProject
local rootProject = root .. locationProject
location(rootProject)
language "C++"

filter "configurations:Debug"
targetdir(pathTargetDebug)
targetname("d_SampleBasic")
filter "configurations:Release"
targetdir(pathTargetRelease)
filter {}

local tempDir = rootProject .. "/temp/"
objdir(tempDir)
kind "SharedLib"

local SourceDir = rootProject .. "/src/";
local IncludeSubDir = rootProject .. "/include/**";
local IncludeDir = rootProject .. "/include/";
dependson { "PluginInteropUnityCUDA" }
files
{
    SourceDir .. "**.c",
    -- SourceThirdPartyGLDir .. "**.c",
    -- ThirdPartyGLDir .. "**.h",
    SourceDir .. "**.cpp",
    IncludeDir .. "**.h",
    IncludeDir .. "**.cuh",
    IncludeDir .. "**.hpp",
    IncludeSubDir .. "**.h",
    IncludeSubDir .. "**.cuh",
    IncludeSubDir .. "**.hpp"
}

includedirs
{
    IncludeDir,
    IncludeSubDir,
    pathThirdPartyGLDir,
    pathUtilitiesInclude,
    pathPluginInteropInclude,
    pathPluginInteropIncludeSubdir,
    pathThirdPartyUnity
}


filter "configurations:Debug"

libdirs
{
    pathTargetDebug
}
links { nameUtilitiesLibDebug, "OpenGL32", namePluginInteropLibDebug }

filter "configurations:Release"

libdirs
{
    pathTargetRelease
}
links { nameUtilitiesLib, "OpenGL32", namePluginInteropLib }

filter {}



-- Add necessary build customization using standard Premake5
-- This assumes you have installed Visual Studio integration for CUDA
-- Here we have it set to 12.2
buildcustomizations "BuildCustomizations/CUDA 12.2"
-- cudaPath "/usr/local/cuda" -- Only affects linux, because the windows builds get CUDA from the VS extension

-- CUDA specific properties
cudaFiles { SourceDir .. "**.cu" }
cudaMaxRegCount "32"

-- Let's compile for all supported architectures (and also in parallel with -t0)
cudaCompilerOptions { "-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
    "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0" }

-- On Windows, the link to cudart is done by the CUDA extension, but on Linux, this must be done manually
if os.target() == "linux" then
    linkoptions { "-L/usr/local/cuda/lib64 -lcudart" }
end

filter "configurations:release"
cudaFastMath "On" -- enable fast math for release
filter ""


-------------------------------
-- [ Utilities ] --
-------------------------------
project "Utilities"
local rootProject = pathUtilities
location(rootProject)
kind "SharedLib"
language "C++"

filter "configurations:Debug"
targetdir(pathTargetDebug)
targetname("d_Utilities")
filter "configurations:Release"
targetdir(pathTargetRelease)
filter {}
local tempDir = rootProject .. "/temp/"
objdir(tempDir)

local SourceDir = rootProject .. "/src/";
local IncludeDir = pathUtilitiesInclude;

-- what files the visual studio project/makefile/etc should know about
files
{
    SourceDir .. "**.c",
    SourceDir .. "**.cpp",
    IncludeDir .. "**.h",
    IncludeDir .. "**.hpp",
}

includedirs
{
    IncludeDir
}

flags { "NoImportLib" }
flags {}

defines { "UTILITIES_SHARED", "UTILITIES_SHARED_EXPORTS" }
symbols "On"

sourceDirAbsolute = path.getabsolute(SourceDir)
objDirAbsolute = path.getabsolute(tempDir)
addCUDAToProject(sourceDirAbsolute, objDirAbsolute)
