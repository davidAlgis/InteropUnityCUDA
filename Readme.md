# Interoperability between Unity Engine and CUDA

This repository shows a demonstration of interoperability between Unity Engine and CUDA. More specifically it proposes to create and render graphics objects (eg. texture) in Unity and editing them directly through CUDA kernel. This permit to bypass compute shader and to benefits of the full capacity of CUDA.


# Plugins

The `Plugin` folder contains the c++ library used for interoperability. They can be regenerated with CMake. Assuming you are at the root of interopUnityCUDA, here are the commands to execute:

```
cd .\Plugin 
mkdir build
cd build
cmake ..
```

Then use your favorite compiler to compile it. On Windows you can use these commands :

```
MsBuild .\PluginInteropUnity.sln /t:rebuild /m /p:Configuration=release 
MsBuild .\PluginInteropUnity.sln /t:rebuild /m /p:Configuration=debug 
```

Moreover, to use the library in you unity project, you need to copy the content of the folder `Debug` and `Release` to your unity project. 

The C++ projet consists of three library : 

## 1. Utilities

This library include a singleton logger to simplify debugging between Unity and Native Plugin. Moreover it contains Unity native plugin API used by the other library.

## 2. PluginInteropUnityCUDA

This library contains the class that handle interoperability between Unity, Graphics API and CUDA. Moreover, it has function to register and call new `Action`. 

An `Action` is a base class from which we can inherits to override functions. These functions will be called on render thread which is a necessary condition to make interoperability works.

## 3. SampleBasic

This library contains two basics examples of actions :
- ActionSampleTexture : it register a Unity __texture__ into CUDA and write some color into it.
- ActionSampleTextureArray : it register a Unity __texture array__ into CUDA and write some color into each texture slice.
- ActionSampleVertexBuffer : it register a Unity __vertex buffer of `float4`__ into CUDA and change their values. 

# InteropUnityCUDA the Unity project

The folder `InteropUnityCUDA` contains the Unity project with the script to handle actions and call them in render thread. Furthermore, there is a script to display in Unity the log informations of the different plugin that use logger of Utilities (see. above).

The project has only one scene that demonstrate the three simple actions describe above. 

# Create your own action

[See the dedicated documentation here.](Plugin/Documentation/CreateAction.md)


# Platform availability

It has been tested only on Unity 2021.1 and CUDA 12.2. At the moment it only work with OpenGL and DirectX11 graphics API.
For DirectX11 texture only works with `Texture2D` type not with `RenderTexture` (see. https://github.com/davidAlgis/InteropUnityCUDA/issues/2).

# Meta

This repository has been developed within the scopes of the thesis of David Algis in collaboration with XLIM and Studio Nyx.
