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
		/// <param name="texture">render texture that will be used in interoperability</param>
		public ActionUnitySampleTextureArray(Texture2DArray texture) : base()
		{
			_actionPtr = createActionSampleTextureArrayBasic(texture.GetNativeTexturePtr(), texture.width, texture.height, texture.depth);
		}
	}

}
