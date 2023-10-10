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
    // this will map the registered texture in cuda
    _texture->mapTextureToSurfaceObject();
    return 0;
}

int Update() override
{       
    // call your kernel here
    // you can write into a texture using `surf2Dwrite` function
    // (see. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=surf2dwrite#surf2dwrite)
    // to get the cudaSurfaceObject_t you can use getter getSurfaceObjectArray()
    return 0;
}

int OnDestroy() override
{
    // unmap it
    _texture->unmapTextureToSurfaceObject();
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
        _texture->mapTextureToSurfaceObject();
        return 0;
    }

    int Update() override
    {       
        // this kernel can be found in SampleBasic project and sample_kernels.cu
        kernelCallerWriteTexture(_texture->getDimGrid(), _texture->getDimBlock(),
                             _texture->getSurfaceObject(), GetTime(),
                             _texture->getWidth(), _texture->getHeight());
        return 0;
    }

    int OnDestroy() override
    {
        _texture->unmapTextureToSurfaceObject(surf);
        _texture->unregisterTextureInCUDA();
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
