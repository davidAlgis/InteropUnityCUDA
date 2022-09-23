require("gmake2")
require("cuda-exported-variables")

local function writeBoolean(property, flag)
    if value == true or value == "On" then
        premake.w(property .. ' += ' .. flag)
    end
end

local function writeBooleanWithFlag(property, flag, value)
    if value == true or value == "On" then
        premake.w(property .. ' += ' .. flag .. '=true')
    elseif value == false or value == "Off" then
        premake.w(property .. ' += ' .. flag .. '=false')
    end
end

local function writeStringMacro(property, value)
    if value ~= nil and value ~= '' then
        premake.w(property .. ' := ' .. value)
    end
end

local function writeTableAsOneString(property, values)
    if values ~= nil then
        writeStringMacro(property, table.concat(values, ' '))
    end
end

local function inlineFileWrite(value)
    premake.w('CUOBJECTS += $(OBJDIR)/' .. path.getbasename(value) .. ".cu.o")
    premake.w('CUGENERATED += $(OBJDIR)/' .. path.getbasename(value) .. ".cu.o")
    premake.w('$(OBJDIR)/' .. path.getbasename(value) .. ".cu.o: " .. path.getabsolute(value) .. ' | $(TARGETDIR) $(OBJDIR)')
	premake.w('\t@echo $(notdir $<)')
    premake.w('\t$(SILENT) $(NVCC) $(NVCCCOMPILEFLAGS) $(FORCE_INCLUDE) -o "$@" -MP -MMD -MF "$(@:%%.cu.o=%%.cu.d)" -c "$<" ')
    premake.w()
end

local function checkForGlob(value)
    matchingFiles = os.matchfiles(value)
    if matchingFiles ~= null then
        table.foreachi(matchingFiles, inlineFileWrite)
    end
end

local function addCUDAFiles(cfg)
    if cfg.cudaFiles ~= null then
        local deviceLinkObj = '$(OBJDIR)/cuDeviceLink.cu.o'
        premake.w('OBJECTS += ' .. deviceLinkObj)
        premake.w('GENERATED += ' .. deviceLinkObj)
        premake.w()
        premake.w("CUOBJECTS :=")
        premake.w("CUGENERATED :=")
        premake.w()

        table.foreachi(cfg.cudaFiles, checkForGlob)

        premake.w(deviceLinkObj .. ': $(CUOBJECTS)')
        premake.w('\t$(SILENT) $(NVCC) -dlink $^ $(NVCCLINKFLAGS) -o $@')
        premake.w()

        premake.w('LINKCMD += $(CUOBJECTS)')
        premake.w()
    end
end

premake.override(premake.modules.gmake2.cpp, 'allRules', function(oldfn, cfg)
    oldfn(cfg)
    addCUDAFiles(cfg)
end)

local function addNVCCPath(cfg)
    local nvcc = 'nvcc';
    if cfg.cudaPath ~= null then
        nvcc = cfg.cudaPath .. '/bin/' .. nvcc;
    end
    premake.w('NVCC := ' .. nvcc)
end

local function addCompilerProps(cfg)
    writeTableAsOneString('NVCCCOMPILEFLAGS', cfg.cudaCompilerOptions)
    writeBooleanWithFlag('NVCCCOMPILEFLAGS', '-rdc', cfg.cudaRelocatableCode)
    writeBoolean('NVCCCOMPILEFLAGS', '-ewp', cfg.cudaExtensibleWholeProgram)
    writeBoolean('NVCCCOMPILEFLAGS', '--use_fast_math', cfg.cudaFastMath)

    if cfg.cudaMaxRegCount ~= nil and cfg.cudaMaxRegCount ~= '' then
        premake.w('NVCCCOMPILEFLAGS += -maxrregcount ' .. cfg.cudaMaxRegCount)
    end

    if cfg.cudaVerbosePTXAS ~= nil and cfg.cudaVerbosePTXAS ~= '' then
        premake.w('NVCCCOMPILEFLAGS += -verbose')
    end
end

local function addLinkerProps(cfg)
    writeTableAsOneString('NVCCLINKFLAGS', cfg.cudaLinkerOptions)
end

premake.override(premake.modules.gmake2, 'header', function(oldfn, cfg)
    oldfn(cfg)

    local kind = iif(cfg.project, 'project', 'workspace')

    if kind == 'project' then
        addNVCCPath(cfg)
        addCompilerProps(cfg)
        addLinkerProps(cfg)
        premake.w()
    end
end)
