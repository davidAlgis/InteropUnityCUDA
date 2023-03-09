using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
	public class ActionUnitySampleTexture : ActionUnity
	{
		const string _dllSampleBasic = "SampleBasic";

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionSampleTextureBasic(IntPtr texturePtr, int width, int height);
		
		/// <summary>
		/// create a pointer to actionSampleTexture object that has been created in native plugin
		/// </summary>
		/// <param name="texture">texture that will be used in interoperability</param>
		public ActionUnitySampleTexture(Texture texture) : base()
		{
			_actionPtr = createActionSampleTextureBasic(texture.GetNativeTexturePtr(), texture.width, texture.height);
		}
	}

}
