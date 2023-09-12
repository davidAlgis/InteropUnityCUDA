using System.Collections;
using System.Collections.Generic;
using ActionUnity;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools;

public class InteropTests 
{
    private GameObject _gameObjectWithInteropHandler;
    private InteropHandlerSample _interopHandlerSample;
    [SetUp]
    public void SetUp()
    {
        _gameObjectWithInteropHandler = new GameObject("TestObject");
        _interopHandlerSample = _gameObjectWithInteropHandler.AddComponent<InteropHandlerSample>();
        _gameObjectWithInteropHandler.AddComponent<PluginLoggerReader>();
    }

    [TearDown]
    public void TearDown()
    {
        // Clean up resources and destroy the GameObject
        Object.Destroy(_gameObjectWithInteropHandler);
    }

    [UnityTest]
    public IEnumerator TestInteropHandler()
    {
        // Wait for a few seconds to allow the simulation to run
        float simulationTime = 0.5f; 
        yield return new WaitForSeconds(simulationTime);

        // Now that the simulation has run, run your tests
        // Yield one more frame to ensure everything is updated
        yield return null; 
           // Perform your tests as previously described
        TextureContainsExpectedValues();
        TextureArrayContainsExpectedValues();
        // ComputeBufferContainsExpectedValues();
    }

public void TextureContainsExpectedValues()
{
    Texture2D originalTexture = _interopHandlerSample.Texture;

    // Create a temporary RenderTexture with the same dimensions as the original texture
    RenderTexture tempRenderTexture = new RenderTexture(originalTexture.width, originalTexture.height, 0);
    RenderTexture.active = tempRenderTexture;

    // Copy the content of the original Texture2D to the RenderTexture
    Graphics.Blit(originalTexture, tempRenderTexture);

    // Create a new Texture2D to read the pixels from the RenderTexture
    Texture2D copiedTexture = new Texture2D(originalTexture.width, originalTexture.height);
    copiedTexture.ReadPixels(new Rect(0, 0, originalTexture.width, originalTexture.height), 0, 0);
    copiedTexture.Apply();

    // Loop through each pixel and check the value
    Color[] pixels = copiedTexture.GetPixels();
    foreach (Color pixel in pixels)
    {
        // Implement your pixel value verification logic here
        float expectedValue = math.abs(math.cos(Time.time));
        Assert.IsTrue(math.abs(expectedValue - pixel.g) < 1e-2f);
    }

    // Clean up resources
    RenderTexture.active = null;
    Object.Destroy(copiedTexture);
    Object.Destroy(tempRenderTexture);
}

    public void TextureArrayContainsExpectedValues()
    {
        Texture2DArray textureArray = _interopHandlerSample.TextureArray;

        // Create a RenderTexture to copy the Texture2DArray.
        RenderTexture renderTexture = new RenderTexture(textureArray.width, textureArray.height, 0, RenderTextureFormat.ARGB32);
        renderTexture.enableRandomWrite = true;
        renderTexture.Create();

        // Set up a temporary camera to render the Texture2DArray to the RenderTexture.
               // Create a temporary camera GameObject and set its target texture.
        GameObject tempCameraObject = new GameObject("TempCamera");
        Camera tempCamera = tempCameraObject.AddComponent<Camera>();
        tempCamera.targetTexture = renderTexture;
        tempCamera.RenderWithShader(Shader.Find("Unlit/Texture"), "RenderType");

        // Create a temporary texture to read the pixels from the RenderTexture.
        Texture2D tempTexture = new Texture2D(textureArray.width, textureArray.height, TextureFormat.ARGB32, false);

        // Read the pixels from the RenderTexture into the temporary texture.
        RenderTexture.active = renderTexture;
        tempTexture.ReadPixels(new Rect(0, 0, textureArray.width, textureArray.height), 0, 0);
        tempTexture.Apply();

        // Loop through the slices of the Texture2DArray and compare pixel values.
        for (int z = 0; z < textureArray.depth; z++)
        {
            for (int x = 0; x < textureArray.width; x++)
            {
                for (int y = 0; y < textureArray.height; y++)
                {
                    Color expectedColor = new Color(z % 2, math.abs((z + 1) * math.cos(Time.time)), 0, 1.0f);
                    Color actualColor = tempTexture.GetPixel(x, y);
                    Debug.Log(expectedColor + " " + actualColor);
                    // Compare the colors with a tolerance.
                    Assert.IsTrue((Vector4.Distance(expectedColor, actualColor) < 1e-2));
                }
            }
        }

        // Clean up temporary objects.
        Object.Destroy(tempCamera.gameObject);
        Object.Destroy(renderTexture);
        Object.Destroy(tempTexture);

    }


    public void ComputeBufferContainsExpectedValues()
    {
        ComputeBuffer computeBuffer = _interopHandlerSample.ComputeBuffer;

        // Implement your verification logic for the 'ComputeBuffer' here
        Assert.IsNotNull(computeBuffer);

        // Set the buffer data to an array to access the values
        float4[] bufferData = new float4[computeBuffer.count];
        computeBuffer.GetData(bufferData);

        // Loop through the buffer data and verify values at each index
        for (int x = 0; x < bufferData.Length; x++)
        {
            float4 expectedValue = new float4
            (
                math.cos(2 * math.PI * Time.time / x),
                math.sin(2 * math.PI * Time.time / x),
                0.0f,
                1.0f
            );

            // Implement your verification logic for the ComputeBuffer data here
            Assert.AreEqual(expectedValue, bufferData[x]);
        }
    }
}
