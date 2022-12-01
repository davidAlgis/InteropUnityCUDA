# Interoperability between Unity Engine and CUDA

This repository shows a demonstration of interoperability between Unity Engine and CUDA. More specifically it proposes to create and render graphics objects (eg. texture) in Unity and editing them directly through CUDA kernel. This permit to overstep compute shader and to benefits of the full capacity of CUDA.


# Plugins

The folder `Plugin` contains a solution that can be generated for visual studio 2019 using [Premake5](https://premake.github.io/download/) with command :

```
premake5 vs2019
```

The solution contains :

## Utilities

This library include a singleton logger to simplify debugging between Unity and Native Plugin. Moreover it contains Unity native plugin API used by the other library.

## PluginInteropUnityCUDA

This library contains the class that handle interoperability between Unity, Graphics API and CUDA. Moreover, it has function to register and call new `Action`. An `Action` is a base class from which we can inherits to override functions. These functions will be called on render thread which is a necessary condition to make interoperability works.

## SampleBasic

This library contains two basics examples of actions :
- ActionSampleTexture : it register a Unity __texture__ into CUDA and write some color into it.
- ActionSampleVertexBuffer : it register a Unity __vertex buffer of `float4`__ into CUDA and change their values. 

# InteropUnityCUDA

The folder `InteropUnityCUDA` contains the Unity project for Unity 2021 and after which have script to handle actions and call them in render thread. It has only one scene that demonstrate the two simple actions describe above. 



