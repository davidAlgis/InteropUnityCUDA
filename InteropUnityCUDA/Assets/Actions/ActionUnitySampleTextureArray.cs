using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
	public class ActionUnitySampleTextureArray : ActionUnity
	{
		const string _dllSampleBasic = "SampleBasic";

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionSampleTextureArrayBasic(IntPtr texturePtr, int width, int height, int depth);
		
		/// <summary>
		/// create a pointer to actionSampleTextureArray object that has been created in native plugin
		/// </summary>
		/// <param name="renderTexture">render texture that will be used in interoperability</param>
		public ActionUnitySampleTextureArray(RenderTexture renderTexture) : base()
		{
			_actionPtr = createActionSampleTextureArrayBasic(renderTexture.GetNativeTexturePtr(), renderTexture.width, renderTexture.height, renderTexture.volumeDepth);
		}
	}

}
