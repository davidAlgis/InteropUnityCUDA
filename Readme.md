# Interoperability Between Unity Engine and CUDA

This repository demonstrates and gives tools for interoperability between the Unity Engine and CUDA. Specifically, it showcases how to create and render graphical objects (e.g., textures) in Unity and edit them directly through CUDA kernels. This approach bypasses compute shaders, leveraging the full capabilities of CUDA. For more details about this project you can read our article [**_InteropUnityCUDA: A Tool for Interoperability Between Unity and CUDA_**](https://doi.org/10.1002/spe.3414).


## Plugins

The `Plugin` folder contains the C++ libraries used for interoperability. These libraries can be regenerated using CMake. Assuming you are at the root of the `interopUnityCUDA` repository, execute the following commands:

```bash
cd ./Plugin
mkdir build
cmake -B build
```

To compile the libraries:

```bash
cmake --build build --config Release
cmake --build build --config Debug
```

To use the library in your Unity project, copy the contents of the `Debug` and `Release` folders to your Unity project.

The C++ project consists of three libraries:

### 1. Utilities

This library includes a singleton logger to simplify debugging between Unity and the native plugin. It also contains Unity's native plugin API, which is used by the other libraries.

### 2. PluginInteropUnityCUDA

This library provides the classes that handle interoperability between Unity, the graphics API, and CUDA. Additionally, it includes functions to register and invoke new `Action` objects.

An `Action` is a base class from which you can inherit to override specific functions. These functions are executed on the render thread, which is a necessary condition for making interoperability work.

### 3. SampleBasic

This library contains two basic examples of actions:
- **ActionSampleTexture**: Registers a Unity **texture** with CUDA and writes some color data to it.
- **ActionSampleTextureArray**: Registers a Unity **texture array** with CUDA and writes color data to each texture slice.
- **ActionSampleVertexBuffer**: Registers a Unity **vertex buffer of `float4`** with CUDA and modifies its values.

## InteropUnityCUDA Unity Project

The `InteropUnityCUDA` folder contains the Unity project, which includes scripts for handling actions and invoking them on the render thread. Additionally, there is a script for displaying log information in Unity, using the logger from the Utilities library (see above).

The project includes a single scene demonstrating the three basic actions described earlier.

## Generate and Add the InteropUnityCUDA Package to Your Project

[See the dedicated documentation here.](Plugin/Documentation/GenerateUnityPackage.md)

## Create Your Own Action

[See the dedicated documentation here.](Plugin/Documentation/CreateAction.md)

## Platform Availability

This project has been tested with Unity 2021.1 and CUDA 12.2. It currently supports only OpenGL and DirectX11 graphics APIs. The plugin is designed for Windows but can also be compiled for Linux.

### Limitations

- For DirectX11, textures work only with the `Texture2D` type, not with `RenderTexture` ([see issue #2](https://github.com/davidAlgis/InteropUnityCUDA/issues/2)).
- Buffers cannot be written by both Unity **and** CUDA without remapping/unmapping the buffer after each write operation in CUDA ([see issue #12](https://github.com/davidAlgis/InteropUnityCUDA/issues/12)).

## Meta

This repository was developed as part of David Algis' thesis in collaboration with XLIM and Studio Nyx.

If you find this project useful, please consider citing our associated publication:

**Algis, D., Bramas, B., Darles, E., & Aveneau, L. (2025). InteropUnityCUDA: A Tool for Interoperability Between Unity and CUDA. _Software: Practice and Experience._** [https://doi.org/10.1002/spe.3414](https://doi.org/10.1002/spe.3414)

```bibtex
@article{algis2025interopunitycuda,
  author = {Algis, David and Bramas, Berenger and Darles, Emmanuelle and Aveneau, Lilian},
  title = {InteropUnityCUDA: A Tool for Interoperability Between Unity and CUDA},
  journal = {Software: Practice and Experience},
  year = {2025},
  doi = {10.1002/spe.3414},
  url = {https://onlinelibrary.wiley.com/doi/10.1002/spe.3414}
}