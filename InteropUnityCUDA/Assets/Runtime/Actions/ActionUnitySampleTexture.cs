using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
	public class ActionUnitySampleTexture : ActionUnity
	{

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        private const string _dllSampleBasic = "d_SampleBasic";
#else
        private const string _dllSampleBasic = "SampleBasic;
#endif

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
