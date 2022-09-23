-- Variables exported to the user, descriptions in the README

premake.api.register {
    name = "cudaRelocatableCode",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaExtensibleWholeProgram",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaCompilerOptions",
    scope = "config",
    kind = "table"
}

premake.api.register {
    name = "cudaLinkerOptions",
    scope = "config",
    kind = "table"
}

premake.api.register {
    name = "cudaFastMath",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaVerbosePTXAS",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaMaxRegCount",
    scope = "config",
    kind = "string"
}

premake.api.register {
    name = "cudaFiles",
    scope = "config",
    kind = "table"
}

premake.api.register {
    name = "cudaPath",
    scope = "config",
    kind = "string"
}
