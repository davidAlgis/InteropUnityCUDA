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
		
		public ActionUnitySampleTexture(RenderTexture rt) : base()
		{
			_actionPtr = createActionSampleTextureBasic(rt.GetNativeTexturePtr(), rt.width, rt.height);
		}
	}

}
