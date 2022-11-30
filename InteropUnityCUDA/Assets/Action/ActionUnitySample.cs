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

		[DllImport(_dllSampleBasic)]
		private static extern void setTextureActionToto(IntPtr actionPtr, IntPtr texturePtr);


		public ActionUnitySample(RenderTexture rt) : base()
		{
			_actionPtr = createActionSampleBasic(rt.GetNativeTexturePtr(), rt.width, rt.height);
		}

		public void SetTextureActionToto(IntPtr texturePtr)
		{
			setTextureActionToto(_actionPtr, texturePtr);
		}

	}

}
