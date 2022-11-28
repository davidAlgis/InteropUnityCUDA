-- For documentation on all premake functions please see the wiki:
--   https://github.com/premake/premake-core/wiki

local ROOT = "./"
local PATH_TARGET = "../InteropUnityCUDA/Assets/Plugin"
local PATH_SAMPLE_PROJECT = "SampleBasic"
local PATH_UTILITIES = ROOT .. "Utilities"
local PATH_PLUGIN_INTEROP = ROOT .. "PluginInteropUnityCUDA"
local NAME_UTILITIES_LIB = "Utilities.lib"
local NAME_PLUGIN_INTEROP_LIB = "PluginInteropUnityCUDA.lib"
local PATH_UTILITIES_INCLUDE = PATH_UTILITIES .. "/include/"
local PATH_PLUGIN_INTEROP_INCLUDE = PATH_PLUGIN_INTEROP .. "/include/"
local PATH_PLUGIN_INTEROP_INCLUDE_SUBDIR = PATH_PLUGIN_INTEROP_INCLUDE .. "**"
local PATH_UTILITIES_THIRD_PARTY_UNITY = PATH_UTILITIES .. "/thirdParty/unity/include/"

require('premake5-cuda')
---------------------------------
-- [ WORKSPACE CONFIGURATION   --
---------------------------------
-- Setting up the workspace. A workspace can have multiple Projects. 
-- In visual studio a workspace corresponds to a solution.
workspace "Plugin"

    configurations { "Debug", "Release" } -- Optimization/General config mode in VS
    platforms      { "x64"}            -- Dropdown platforms section in VS
    -------------------------------
    -- [ COMPILER/LINKER CONFIG] --
    -------------------------------
    -- Here we are setting up what differentiates the configurations that we called "Debug" and "Release"
    filter "configurations:Debug"    defines { "DEBUG" }  symbols  "On"
    filter "configurations:Release"  defines { "NDEBUG" } optimize "On"
    filter { "platforms:*64" } architecture "x64"

    defines {"PLUGINGLINTEROPCUDA_SHARED_EXPORTS"} symbols  "On"

    -- building makefiles
    filter { "action:gmake" }

    -- building make files on mac specifically
    filter { "system:macosx", "action:gmake"}
        toolset "clang"
        -- Toolset changes the compiler/linker/etc of the current project configuration.
        --     "clang" is used in this case, because premake defaults to using gcc/g++ 
        --     when used with the 'gmake' action, but macosx ships with clang by default
        --     and if we don't set it explicitly we will get some warnings.

    filter {}  -- clear filter when you know you no longer need it!

-------------------------------
-- [ PluginGLInteropCUDA ] --
------------------------------- 
project "PluginInteropUnityCUDA"
    local ROOT_PROJECT = PATH_PLUGIN_INTEROP
    location (ROOT_PROJECT)
    language "C++"
    targetdir (PATH_TARGET, "SampleBasic")
    objdir (ROOT_PROJECT .. "/temp/")
    kind "SharedLib" 
    
    local SourceDir = ROOT_PROJECT .. "/src/";
    local IncludeSubDir = PATH_PLUGIN_INTEROP_INCLUDE_SUBDIR;
    local IncludeDir = PATH_PLUGIN_INTEROP_INCLUDE;
    local ThirdPartyGLDir = ROOT_PROJECT .. "/thirdParty/gl3w/include/";
    local SourceThirdPartyGLDir = ROOT_PROJECT .. "/thirdParty/gl3w/src/";

    files
    {
        SourceDir .. "**.c",
        SourceThirdPartyGLDir .. "**.c",
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
        ThirdPartyGLDir,
        PATH_UTILITIES_INCLUDE,
        PATH_UTILITIES_THIRD_PARTY_UNITY
    }


    libdirs
    {
        PATH_TARGET
    }

    filter { "system:windows" }
        links{NAME_UTILITIES_LIB, "OpenGL32"}

    filter { "system:not windows" }
        links { "GL" , NAME_UTILITIES_LIB}

    filter {}


    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.4
    buildcustomizations "BuildCustomizations/CUDA 11.7"
    cudaPath "/usr/local/cuda" -- Only affects linux, because the windows builds get CUDA from the VS extension

    -- CUDA specific properties
    cudaFiles 
    {
        SourceDir .. "**.cu"
    }
    cudaMaxRegCount "32"

    -- Let's compile for all supported architectures (and also in parallel with -t0)
    cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
                         "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                         "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
                         "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"}                      

    -- On Windows, the link to cudart is done by the CUDA extension, but on Linux, this must be done manually
    if os.target() == "linux" then 
        linkoptions {"-L/usr/local/cuda/lib64 -lcudart"}
    end

    filter "configurations:release"
    cudaFastMath "On" -- enable fast math for release
    filter ""




-------------------------------
-- [ SampleBasic ] --
------------------------------- 
project "SampleBasic"
    local LOCATION_PROJECT = PATH_SAMPLE_PROJECT
    local ROOT_PROJECT = ROOT .. LOCATION_PROJECT
    location (ROOT_PROJECT)
    language "C++"
    targetdir (PATH_TARGET)
    objdir (ROOT_PROJECT .. "/temp/")
    kind "SharedLib" 
    
    local SourceDir = ROOT_PROJECT .. "/src/";
    local IncludeSubDir = ROOT_PROJECT .. "/include/**";
    local IncludeDir = ROOT_PROJECT .. "/include/";
    local ThirdPartyGLDir = ROOT_PROJECT .. "/thirdParty/gl3w/include/";
    local SourceThirdPartyGLDir = ROOT_PROJECT .. "/thirdParty/gl3w/src/";

    files
    {
        SourceDir .. "**.c",
        SourceThirdPartyGLDir .. "**.c",
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
        ThirdPartyGLDir,
        PATH_UTILITIES_INCLUDE,
        PATH_PLUGIN_INTEROP_INCLUDE,
        PATH_PLUGIN_INTEROP_INCLUDE_SUBDIR,
        PATH_UTILITIES_THIRD_PARTY_UNITY
    }


    libdirs
    {
        PATH_TARGET
    }

    --here we need the dll INTEROP too
    filter { "system:windows" }
        links{NAME_UTILITIES_LIB, "OpenGL32", NAME_PLUGIN_INTEROP_LIB}

    filter { "system:not windows" }
        links { "GL" , NAME_UTILITIES_LIB}

    filter {}


    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.4
    buildcustomizations "BuildCustomizations/CUDA 11.7"
    cudaPath "/usr/local/cuda" -- Only affects linux, because the windows builds get CUDA from the VS extension

    -- CUDA specific properties
    cudaFiles 
    {
        SourceDir .. "**.cu"
    }
    cudaMaxRegCount "32"

    -- Let's compile for all supported architectures (and also in parallel with -t0)
    cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
                         "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                         "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
                         "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"}                      

    -- On Windows, the link to cudart is done by the CUDA extension, but on Linux, this must be done manually
    if os.target() == "linux" then 
        linkoptions {"-L/usr/local/cuda/lib64 -lcudart"}
    end

    filter "configurations:release"
    cudaFastMath "On" -- enable fast math for release
    filter ""


-------------------------------
-- [ Utilities ] --
------------------------------- 
project "Utilities"
    local ROOT_PROJECT = PATH_UTILITIES
    location (ROOT_PROJECT)
    kind "SharedLib" 
    language "C++"
    targetdir (PATH_TARGET)
    objdir (ROOT_PROJECT .. "/temp/".. LOCATION_PROJECT)
    
    local SourceDir = ROOT_PROJECT .. "/src/";
    local IncludeDir = PATH_UTILITIES_INCLUDE;
    local ThirdPartyUnityDir = ROOT_PROJECT .. "/thirdParty/unity/include/";
    
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
        IncludeDir,
        ThirdPartyUnityDir
    }

    flags { "NoImportLib" }
    flags {}

    defines {"UTILITIES_SHARED", "UTILITIES_SHARED_EXPORTS"} symbols  "On"



