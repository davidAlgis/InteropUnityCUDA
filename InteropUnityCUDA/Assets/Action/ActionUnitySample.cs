using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{

	public class ActionUnitySample : ActionUnity
	{
		const string _dllSampleBasic = "SampleBasic";

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionSampleBasic(IntPtr texturePtr, int width, int height);
		
		public ActionUnitySample(RenderTexture rt) : base()
		{
			_actionPtr = createActionSampleBasic(rt.GetNativeTexturePtr(), rt.width, rt.height);
		}
	}

}
