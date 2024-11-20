# Create Your Own Action

## Native Plugin Code

To create your own `Action` and benefit from CUDA and interoperability, follow these steps:

1. Include `PluginInteropUnityCUDA` in your library.
2. Create a class (in this example, `MyAction`) that derives from `Action` in `Action.h`:

```cpp
#include "action.h"

class MyAction : public Action {
```

3. Create a constructor that takes the necessary parameters (e.g., a pointer to a Unity texture and the texture's resolution):

```cpp
#include "unityPlugin.h"
#include "texture.h"

MyAction(void* textureUnityPtr, int resolutionTexture) : Action() 
{
    _resolutionTexture = resolutionTexture;
    // Create a Texture object that can be registered/mapped to CUDA
    _texture = CreateTextureInterop(textureUnityPtr, resolutionTexture, resolutionTexture);
}
```

4. Define a function to be called from Unity to get a pointer to `MyAction` and register it in `PluginInteropUnityCUDA`:

```cpp
// for mangling
extern "C" {
    // Enables Unity C# to create and interact with MyAction
    UNITY_INTERFACE_EXPORT MyAction* UNITY_INTERFACE_API 
    createMyAction(void* textureUnityPtr, int resolutionTexture)
    {
        return (new MyAction(textureUnityPtr, resolutionTexture));            
    }
}
```

5. Override the functions `Start()`, `Update()`, and `OnDestroy()` as needed. In this example, we register and map the texture in CUDA and call a simple kernel:

```cpp
int Start() override
{
    // Perform one-time setup at start
    _texture->registerTextureInCUDA();
    _texture->mapTextureToSurfaceObject();
    return 0;
}

int Update() override
{       
    // Call your kernel here
    // Use the `surf2Dwrite` function to write into the texture
    // (see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=surf2dwrite#surf2dwrite)
    // Use getSurfaceObjectArray() to get the cudaSurfaceObject_t
    return 0;
}

int OnDestroy() override
{
    // Unmap and unregister the texture
    _texture->unmapTextureToSurfaceObject();
    _texture->unRegisterTextureInCUDA();
    return 0;
}
```

### Complete Example Code

```cpp
#include "action.h"

class MyAction : public Action {
public:
    // Constructor with initialization arguments, such as a Unity texture pointer
    MyAction(void* textureUnityPtr, int resolutionTexture) : Action() 
    {
        _resolutionTexture = resolutionTexture;
        // Create a Texture object that can be registered/mapped to CUDA
        _texture = CreateTextureInterop(textureUnityPtr, resolutionTexture, resolutionTexture);
    }

    int Start() override
    {
        _texture->registerTextureInCUDA();
        _texture->mapTextureToSurfaceObject();
        return 0;
    }

    int Update() override
    {       
        // Example kernel call (see SampleBasic project and sample_kernels.cu)
        kernelCallerWriteTexture(
            _texture->getDimGrid(), 
            _texture->getDimBlock(),
            _texture->getSurfaceObject(), 
            GetTime(),
            _texture->getWidth(), 
            _texture->getHeight()
        );
        return 0;
    }

    int OnDestroy() override
    {
        _texture->unmapTextureToSurfaceObject();
        _texture->unRegisterTextureInCUDA();
        return 0;
    }

private:
    Texture* _texture;
    int _resolutionTexture;
};

extern "C" {
    UNITY_INTERFACE_EXPORT MyAction* UNITY_INTERFACE_API 
    createMyAction(void* textureUnityPtr, int resolutionTexture)
    {
        return (new MyAction(textureUnityPtr, resolutionTexture));            
    }
}
```

---

## Unity Code

To integrate your action in Unity and register it with the `PluginInteropUnityCUDA`, follow these steps:

6. Create a C# class (e.g., `MyActionUnity`) derived from `ActionUnity` in `ActionUnity.cs`. This class represents your native `MyAction` object in Unity.

7. Import the function created in step 4 (e.g., `createMyAction`) into your Unity C# code.

8. Set the `_actionPtr` member to the return value of the imported function (e.g., `MyAction*`):

```csharp
public class MyActionUnity : ActionUnity
{
    const string _myDllName = "MyDllName"; // Specify your DLL name

    [DllImport(_myDllName)]
    private static extern IntPtr createMyAction(IntPtr textureUnityPtr, int resolutionTexture);

    // Constructor to create the MyAction object
    public MyActionUnity(RenderTexture rt) : base()
    {
        // Set the pointer to the native MyAction object
        _actionPtr = createMyAction(rt.GetNativeTexturePtr(), rt.width);
    }
}
```

9. Create another C# class (e.g., `MyInteropHandler`) derived from `InteropHandler` in `InteropHandler.cs`. This class will handle the Unity-side initialization and registration of your action.

10. Override the `InitializeActions()` function to construct your action, register it with `PluginInteropUnityCUDA`, and call its `Start()` function:

```csharp
protected override void InitializeActions()
{
    // Create a RenderTexture
    RenderTexture _renderTexture = new RenderTexture(...);

    // Construct the action
    MyActionUnity myAction = new MyActionUnity(_renderTexture);

    // Register the action with PluginInteropUnityCUDA
    RegisterActionUnity(myAction, "myRegistrationKey");

    // Call the overridden Start function
    CallActionStart("myRegistrationKey");
}
```

11. Override the `CallUpdateActions()` function to call the overridden `Update()` function:

```csharp
protected override void CallUpdateActions()
{
    // Call the overridden Update function
    CallActionUpdate("myRegistrationKey");
}
```

12. Override the `CallOnDestroy()` function to call the overridden `OnDestroy()` function:

```csharp
protected override void CallOnDestroy()
{
    // Call the overridden OnDestroy function
    CallActionOnDestroy("myRegistrationKey");
}
```