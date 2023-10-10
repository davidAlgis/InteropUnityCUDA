using System.Collections;
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
    public IEnumerator TestTextureInteropHandler()
    {
        // Wait for a few seconds to allow the simulation to run
        var simulationTime = 1.0f;
        yield return new WaitForSeconds(simulationTime);

        // Now that the simulation has run, run your tests
        // Yield one more frame to ensure everything is updated
        yield return null;
        // Perform your tests as previously described
        TextureContainsExpectedValues();
    }

    [UnityTest]
    public IEnumerator TestTextureArrayInteropHandler()
    {
        // Wait for a few seconds to allow the simulation to run
        var simulationTime = 1.0f;
        yield return new WaitForSeconds(simulationTime);

        // Now that the simulation has run, run your tests
        // Yield one more frame to ensure everything is updated
        yield return null;
        TextureArrayContainsExpectedValues();
    }

    [UnityTest]
    public IEnumerator TestVertexBufferInteropHandler()
    {
        // Wait for a few seconds to allow the simulation to run
        var simulationTime = 1.0f;
        yield return new WaitForSeconds(simulationTime);

        // Now that the simulation has run, run your tests
        // Yield one more frame to ensure everything is updated
        yield return null;
        ComputeBufferVertexContainsExpectedValues();
    }

    [UnityTest]
    public IEnumerator TestStructBufferInteropHandler()
    {
        // Wait for a few seconds to allow the simulation to run
        var simulationTime = 1.0f;
        yield return new WaitForSeconds(simulationTime);

        // Now that the simulation has run, run your tests
        // Yield one more frame to ensure everything is updated
        yield return null;
        ComputeBufferStructContainsExpectedValues();
    }

    public void TextureContainsExpectedValues()
    {
        var originalTexture = _interopHandlerSample.Texture;

        // Create a temporary RenderTexture with the same dimensions as the original texture
        RenderTexture tempRenderTexture = new(originalTexture.width, originalTexture.height, 0);
        RenderTexture.active = tempRenderTexture;

        // Copy the content of the original Texture2D to the RenderTexture
        Graphics.Blit(originalTexture, tempRenderTexture);

        // Create a new Texture2D to read the pixels from the RenderTexture
        Texture2D copiedTexture = new(originalTexture.width, originalTexture.height);
        copiedTexture.ReadPixels(new Rect(0, 0, originalTexture.width, originalTexture.height), 0, 0);
        copiedTexture.Apply();

        // Loop through each pixel and check the value
        var pixels = copiedTexture.GetPixels();
        foreach (var pixel in pixels)
        {
            // Implement your pixel value verification logic here
            var expectedValue = math.abs(math.cos(Time.time/10.0f));
            bool isExepected = math.abs(expectedValue - pixel.g) < 1e-1f;
            if(isExepected == false)
            {
                Debug.LogError(expectedValue + " was expected but read " + pixel.g);
            } 
            Assert.IsTrue(isExepected);
        }

        // Clean up resources
        RenderTexture.active = null;
        Object.Destroy(copiedTexture);
        Object.Destroy(tempRenderTexture);
    }

    public void TextureArrayContainsExpectedValues()
    {
        var textureArray = _interopHandlerSample.TextureArray;
        Debug.LogWarning("TODO implement this function");
        // TODO to implement
    }

    public void ComputeBufferVertexContainsExpectedValues()
    {
        var computeBuffer = _interopHandlerSample.ComputeVertexBuffer;

        // Implement your verification logic for the 'ComputeBuffer' here
        Assert.IsNotNull(computeBuffer);

        // Set the buffer data to an array to access the values
        var bufferData = new float4[computeBuffer.count];
        computeBuffer.GetData(bufferData);

        // Loop through the buffer data and verify values at each index
        for (var x = 0; x < bufferData.Length; x++)
        {
            float4 expectedValue = new(
                math.cos(2 * math.PI * Time.time / (math.abs(x) + 1.0f)),
                math.sin(2 * math.PI * Time.time / (math.abs(x) + 1.0f)),
                0.0f,
                1.0f
            );

            Assert.IsTrue(math.length(expectedValue - bufferData[x]) < 1e-2f);
        }
    }


    public void ComputeBufferStructContainsExpectedValues()
    {
        var computeBuffer = _interopHandlerSample.ComputeStructBuffer;

        // Implement your verification logic for the 'ComputeBuffer' here
        Assert.IsNotNull(computeBuffer);

        // Set the buffer data to an array to access the values
        var bufferData = new InteropHandlerSample.SampleStructInterop[computeBuffer.count];
        computeBuffer.GetData(bufferData);

        // Loop through the buffer data and verify values at each index
        for (var x = 0; x < bufferData.Length; x++)
        {
            var expectedValueX = math.cos(Time.time) / (math.abs(x) + 1.0f);
            var expectedValueN = x * (int)Time.time;

            Assert.IsTrue(math.abs(expectedValueX - bufferData[x].x) < 1e-2f);
            Assert.IsTrue(expectedValueN - bufferData[x].n == 0);
        }
    }
}