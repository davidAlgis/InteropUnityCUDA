# Interoperability between Unity Engine and CUDA

This repository shows a demonstration of interoperability between Unity Engine and CUDA. More specifically it proposes to create and render graphics objects (eg. texture) in Unity and editing them directly through CUDA kernel. This permit to bypass compute shader and to benefits of the full capacity of CUDA.

# Description

## Plugins

The folder `Plugin` contains a solution that can be generated for visual studio 2019 using [Premake5](https://premake.github.io/download/) with command :

```
premake5 vs2019
```

The solution contains :

### Utilities

This library include a singleton logger to simplify debugging between Unity and Native Plugin. Moreover it contains Unity native plugin API used by the other library.

### PluginInteropUnityCUDA

This library contains the class that handle interoperability between Unity, Graphics API and CUDA. Moreover, it has function to register and call new `Action`. 

An `Action` is a base class from which we can inherits to override functions. These functions will be called on render thread which is a necessary condition to make interoperability works.

### SampleBasic

This library contains two basics examples of actions :
- ActionSampleTexture : it register a Unity __texture__ into CUDA and write some color into it.
- ActionSampleVertexBuffer : it register a Unity __vertex buffer of `float4`__ into CUDA and change their values. 

## InteropUnityCUDA the Unity project

The folder `InteropUnityCUDA` contains the Unity project with the script to handle actions and call them in render thread. Furthermore, there is a script to display in Unity the log informations of the different plugin that use logger of Utilities (see. above).

The project has only one scene that demonstrate the two simple actions describe above. 

# Create your own action

## Native plugin code

To program your own `Action` and benefit of CUDA and interoperability you have to :
1. include `PluginInteropUnityCuda` into your library ;
2. create a class (in our example `MyAction`) that's derived from `Action` in `Action.h` ;

```
#include "action.h"

class MyAction: public Action {
```

3. create a constructor that will have the necessary information (in our example a pointer toward a unity texture and the resolution of this texture) ;

```
#include "unityPlugin.h"
#include "texture.h"

MyAction(void* textureUnityPtr, int resolutionTexture) : Action() 
{
    _resolutionTexture = resolutionTexture;
    // Create an object Texture that can be register/map to CUDA
    _texture = CreateTextureInterop(texturePtr
        , resolutionTexture, resolutionTexture);
}
```

4. create a function that will be call in Unity to get a pointer on `MyAction` to register it in `PluginInteropUnityCUDA` ;

```
// for mangling
extern "C" {
    // to make possible to create and have MyAction in Unity C#
    UNITY_INTERFACE_EXPORT MyAction* UNITY_INTERFACE_API 
        createMyAction(void* textureUnityPtr, int resolutionTexture)
    {
        return (new MyAction(textureUnityPtr, resolutionTexture));            
    }
}
```
5. Override the functions `Start()`, `Update()` and `OnDestroy()` in function of your purpose (in our example it only register and map the texture in CUDA call a simple kernel) 


```
int Start() override
{
    // this has to be done once at start
    _texture->registerTextureInCUDA();
    return 0;
}

int Update() override
{       
    // this will map the registered texture in cuda
    cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject();
    // write into it
    _texture->writeTexture(surf, GetTime());
    // unmap it
    _texture->unMapTextureToSurfaceObject(surf);

    // call interoperability function here
    // call your kernel here
    return 0;
}

int OnDestroy() override
{
    _texture->unRegisterTextureInCUDA();
    return 0;
}
```

Here is the complete code :

```
#include "action.h"

class MyAction: public Action {
public:
    // Some arguments to initialize my action, for 
    // example a pointer toward a texture from Unity, etc.
    MyAction(void* textureUnityPtr, int resolutionTexture) : Action() 
    {
        _resolutionTexture = resolutionTexture;
        // Create an object Texture that can be register/map to CUDA
        _texture = CreateTextureInterop(texturePtr
            , resolutionTexture, resolutionTexture);
    }

    int Start() override
    {
        _texture->registerTextureInCUDA();
        return 0;
    }

    int Update() override
    {       
        cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject();
        _texture->writeTexture(surf, GetTime());
        _texture->unMapTextureToSurfaceObject(surf);
        return 0;
    }

    int OnDestroy() override
    {
        _texture->unRegisterTextureInCUDA();
        return 0;
    }

private:
    // Some attributes that I will use in my action
    Texture* _texture;
    int _resolutionTexture;
};

extern "C" {

    UNITY_INTERFACE_EXPORT MyAction* UNITY_INTERFACE_API 
        createMyAction(void* textureUnityPtr, int resolutionTexture)
    {
        // to make possible to create and have MyAction in Unity C#
        return (new MyAction(textureUnityPtr, resolutionTexture));            
    }
}

```

## Unity code

To plug your action in Unity and register it in plugin `PluginInteropUnityCUDA` you need to : 

6. create a C# class (in our example `MyActionUnity`) that's derived from `ActionUnity` in `ActionUnity.cs`. This class carry the informations of our action in the native code (in our example `MyAction`) ;
7. import the function that was created above in step 4. (in our example `createMyAction`);
8. set the member `_actionPtr` to the return type of the imported function (in our example `MyAction*`);  

```
public class MyActionUnity : ActionUnity
{
    // Here put the dll name
    const string _myDllName = "MyDllName";

    [DllImport(_myDllName)]
    private static extern IntPtr createMyAction(IntPtr textureUnityPtr, int resolutionTexture);

    // we create the object MyAction in constructor of MyActionUnity
    public MyActionUnity(RenderTexture rt) : base()
    {
        // the pointer toward our object MyAction is set in _actionPtr
        _actionPtr = createMyAction(rt.GetNativeTexturePtr(), rt.width);
    }
}
```

9. create a C# class (in our example `MyInteropHandler`) that's derived from `InteropHandler` in `InteropHandler.cs`. This class will create our action in Unity (in our example `MyActionUnity.cs`) and it'll call the function of the plugin `PluginInteropUnityCUDA` to register and call the action ;

10. override the function `InitializeActions()` to initialize the actions in `PluginInteropUnityCUDA`, construct your action with the constructor of step 8. register it with function `RegisterActionUnity` and by defining a registrationKey and call the function `CallActionStart` to call the `Start` function has been vo.

```
protected override void InitializeActions()
{
    // create a render texture in _renderTexture

    // construct an action using constructor of step 8.
    MyAction myAction = new MyAction(_renderTexture);
    // register the action in PluginInteropUnityCyda
    RegisterActionUnity(myAction, "myRegistrationKey");
    // we call the start function that was override in step 5
    CallActionStart("myRegistrationKey");
}
```

11. override the function `CallUpdateActions` to call the `Update` function override in step 5 ;


```
protected override void CallUpdateActions()
{
    // we call the update function that was override in step 5
    CallActionUpdate("myRegistrationKey");
}
```

12. override the function `CallOnDestroy()` to call the the `OnDestroy` function override in step 5 ;

```
protected override void CallOnDestroy()
{
    // we call the on destroy function that was override in step 5
    CallActionOnDestroy("myRegistrationKey");
}
```


# Platform availability

It's has been tested only on Unity 2021.1 and CUDA 11.7. At the moment it only work with OpenGL.

# Meta

This repository has been developed within the scopes of the thesis of David Algis in collaboration with XLIM and Studio Nyx.
