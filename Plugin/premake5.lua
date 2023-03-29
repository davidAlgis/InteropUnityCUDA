function loadrequire(module, linkToRepository)
    local function requiref(module)
        require(module)
    end
    res = pcall(requiref,module)
    if not(res) then
        print("Could not find module :"
        , module," it's available at this link : "
        , linkToRepository)
    end
end


local root = "./"
local pathTarget = "../InteropUnityCUDA/Assets/Plugin"
local pathSampleProject = "SampleBasic"
local pathUtilities = root .. "Utilities"
local pathPluginInterop = root .. "PluginInteropUnityCUDA"
local nameUtilitiesLib = "Utilities.lib"
local namePluginInteropLib = "PluginInteropUnityCUDA.lib"
local pathUtilitiesInclude = pathUtilities .. "/include/"
local pathPluginInteropInclude = pathPluginInterop .. "/include/"
local pathPluginInteropIncludeSubdir = pathPluginInteropInclude .. "**"
local pathThirdPartyUnity = pathPluginInterop .. "/thirdParty/unity/include/"

loadrequire('premake5-cuda', "https://github.com/theComputeKid/premake5-cuda")
print("We use export compile commands module to export compilation database for clang. If you don't have the module you won't be able to compile with clang, BUT you can still compile with visual studio !")
loadrequire('export-compile-commands', "https://github.com/null-black/premake-export-compile-commands")
---------------------------------
-- [ WORKSPACE CONFIGURATION   --
---------------------------------
-- Setting up the workspace. A workspace can have multiple Projects. 
-- In visual studio a workspace corresponds to a solution.
workspace "PluginInteropUnity"

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
    local rootProject = pathPluginInterop
    location (rootProject)
    language "C++"
    targetdir (pathTarget, "SampleBasic")
    objdir (rootProject .. "/temp/")
    kind "SharedLib" 
    
    local SourceDir = rootProject .. "/src/";
    local IncludeSubDir = pathPluginInteropIncludeSubdir;
    local IncludeDir = pathPluginInteropInclude;
    local ThirdPartyGLDir = rootProject .. "/thirdParty/gl3w/include/";
    local SourceThirdPartyGLDir = rootProject .. "/thirdParty/gl3w/src/";

    dependson {"Utilities"}
    files
    {
        SourceDir .. "**.c",
        SourceThirdPartyGLDir .. "**.c",
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
        ThirdPartyGLDir,
        pathThirdPartyUnity,
        pathUtilitiesInclude,
        pathThirdPartyUnity
    }


    libdirs
    {
        pathTarget
    }

    filter { "system:windows" }
        links{nameUtilitiesLib, "OpenGL32"}

    filter { "system:not windows" }
        links { "GL" , nameUtilitiesLib}

    filter {}


    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.4
    buildcustomizations "BuildCustomizations/CUDA 12.1"
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
    local locationProject = pathSampleProject
    local rootProject = root .. locationProject
    location (rootProject)
    language "C++"
    targetdir (pathTarget)
    objdir (rootProject .. "/temp/")
    kind "SharedLib" 
    
    local SourceDir = rootProject .. "/src/";
    local IncludeSubDir = rootProject .. "/include/**";
    local IncludeDir = rootProject .. "/include/";
    local ThirdPartyGLDir = rootProject .. "/thirdParty/gl3w/include/";
    local SourceThirdPartyGLDir = rootProject .. "/thirdParty/gl3w/src/";
    dependson{"PluginInteropUnityCUDA"}
    files
    {
        SourceDir .. "**.c",
        SourceThirdPartyGLDir .. "**.c",
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
        ThirdPartyGLDir,
        pathUtilitiesInclude,
        pathPluginInteropInclude,
        pathPluginInteropIncludeSubdir,
        pathThirdPartyUnity
    }


    libdirs
    {
        pathTarget
    }

    --here we need the dll INTEROP too
    filter { "system:windows" }
        links{nameUtilitiesLib, "OpenGL32", namePluginInteropLib}

    filter { "system:not windows" }
        links { "GL" , nameUtilitiesLib}

    filter {}


    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.4
    buildcustomizations "BuildCustomizations/CUDA 12.1"
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
    local rootProject = pathUtilities
    location (rootProject)
    kind "SharedLib" 
    language "C++"
    targetdir (pathTarget)
    objdir (rootProject .. "/temp/".. locationProject)
    
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

    defines {"UTILITIES_SHARED", "UTILITIES_SHARED_EXPORTS"} symbols  "On"

    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.4
    buildcustomizations "BuildCustomizations/CUDA 12.1"
    cudaPath "/usr/local/cuda" -- Only affects linux, because the windows builds get CUDA from the VS extension

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


