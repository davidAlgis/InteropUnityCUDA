# Generate the Unity Package

The Unity package associated with `InteropUnityCUDA` can be generated in two ways:

---

## Manual Generation

1. **Compile the Project**  
   If you want the latest version of the code, you need to generate the project and compile it first. Refer to the [dedicated documentation here](Readme.md).  

   Once compiled, copy the results to the Unity folder as follows:

   ```
   [ParentOfInteropUnityCUDA]\InteropUnityCUDA\Plugin\build\Debug ---> [ParentOfInteropUnityCUDA]\InteropUnityCUDA\InteropUnityCUDA\Assets\Runtime\Plugin\Debug
   [ParentOfInteropUnityCUDA]\InteropUnityCUDA\Plugin\build\Release ---> [ParentOfInteropUnityCUDA]\InteropUnityCUDA\InteropUnityCUDA\Assets\Runtime\Plugin\Release
   ```

2. **Prepare the Unity Package Folder**  
   Create a new folder that will contain the Unity package and name it as follows:

   ```
   com.studio-nyx.interop-unity-cuda.<version_number>
   ```

   Replace `<version_number>` with the actual version number of the Unity package.

3. **Create the `package.json` File**  
   Inside this new folder, create a JSON file named `package.json` with the following content. Replace `<version_number>` with the actual version number:

   ```json
   {
       "keywords": [
           "GPU",
           "CUDA",
           "OpenGL",
           "DX11",
           "Native-Plugin",
           "interoperability"
       ],
       "license": "MIT",
       "displayName": "Interop Unity CUDA",
       "name": "com.studio-nyx.interop-unity-cuda",
       "description": "Demonstrates interoperability between Unity Engine and CUDA.",
       "version": "<version_number>",
       "author": {
           "email": "david.algis@tutamail.com",
           "url": "https://github.com/davidAlgis",
           "name": "David Algis"
       },
       "dependencies": {
           "com.unity.mathematics": "1.2.6"
       },
       "unity": "2021.1"
   }
   ```

4. **Copy the Assets**  
   Copy all the content from the `Assets` folder:

   ```
   [ParentOfInteropUnityCUDA]\InteropUnityCUDA\InteropUnityCUDA\Assets\**
   ```

   Paste it into the new folder containing the `package.json`.

---

## Automatic Generation

To generate the package automatically, execute the following script:

```bash
.\packageUnity.py
```

---

## Add the Package to a Unity Project

To add the package to a Unity project:

1. Open your Unity project.
2. Navigate to `Window > Package Manager`.
3. In the new window, click the top-left `+` button.
4. Select **Add Package from Disk...**.

   ![Add Package](addPackage.jpeg)

5. Browse to and select the `package.json` file you created earlier.  
   - If you used the automatic generation, the file is located at:

     ```
     [ParentOfInteropUnityCUDA]\InteropUnityCUDA\Plugin\buildtools\com.studio-nyx.interop-unity-cuda.1.0.1
     ```