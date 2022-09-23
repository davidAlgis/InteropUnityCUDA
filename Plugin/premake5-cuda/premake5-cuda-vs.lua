require("vstudio")
require("cuda-exported-variables")

local function writeBoolean(property, value)
    if value == true or value == "On" then
        premake.w('\t<' .. property .. '>true</' .. property .. '>')
    elseif value == false or value == "Off" then
        premake.w('\t<' .. property .. '>false</' .. property .. '>')
    end
end

local function writeString(property, value)
    if value ~= nil and value ~= '' then
        premake.w('\t<' .. property .. '>' .. value .. '</' .. property .. '>')
    end
end

local function writeTableAsOneString(property, values)
    if values ~= nil then
        writeString(property, table.concat(values, ' '))
    end
end

local function addCompilerProps(cfg)
    premake.w('<CudaCompile>')

    -- Determine architecture to compile for
    if cfg.architecture == "x86_64" or cfg.architecture == "x64" then
        premake.w('\t<TargetMachinePlatform>64</TargetMachinePlatform>')
    elseif cfg.architecture == "x86" then
        premake.w('\t<TargetMachinePlatform>32</TargetMachinePlatform>')
    else
        error("Unsupported Architecture")
    end

    -- Set XML tags to their requested values 
    premake.w('<CodeGeneration></CodeGeneration>')
    writeBoolean('GenerateRelocatableDeviceCode', cfg.cudaRelocatableCode)
    writeBoolean('ExtensibleWholeProgramCompilation', cfg.cudaExtensibleWholeProgram)
    writeBoolean('FastMath', cfg.cudaFastMath)
    writeBoolean('PtxAsOptionV', cfg.cudaVerbosePTXAS)
    writeTableAsOneString('AdditionalOptions', cfg.cudaCompilerOptions)
    writeString('MaxRegCount', cfg.cudaMaxRegCount)

    premake.w('</CudaCompile>')
end

local function addLinkerProps(cfg)
    if cfg.cudaLinkerOptions ~= nil then
        premake.w('<CudaLink>')
        writeTableAsOneString('AdditionalOptions', cfg.cudaLinkerOptions)
        premake.w('</CudaLink>')
    end
end

premake.override(premake.vstudio.vc2010.elements, "itemDefinitionGroup", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCompilerProps)
    table.insert(items, addLinkerProps)
    return items
end)

local function inlineFileWrite(value)
    premake.w('\t<CudaCompile ' .. 'Include=' .. string.escapepattern('"') .. path.getabsolute(value) ..
                  string.escapepattern('"') .. '/>')
end

local function checkForGlob(value)
    matchingFiles = os.matchfiles(value)
    if matchingFiles ~= null then
        table.foreachi(matchingFiles, inlineFileWrite)
    end
end

local function addCUDAFiles(cfg)
    if cfg.cudaFiles ~= null then
        premake.w('<ItemGroup>')
        table.foreachi(cfg.cudaFiles, checkForGlob)
        premake.w('</ItemGroup>')
    end
end

premake.override(premake.vstudio.vc2010.elements, "project", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCUDAFiles)
    return items
end)
