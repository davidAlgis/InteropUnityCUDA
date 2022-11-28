using System;
using System.Runtime.InteropServices;

namespace ActionUnity
{

	public class ActionUnitySample : ActionUnity
	{
		const string _dllSampleBasic = "SampleBasic";

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionToto();

		[DllImport(_dllSampleBasic)]
		private static extern void setTextureActionToto(IntPtr actionPtr, IntPtr texturePtr);


		public ActionUnitySample() : base()
		{
			_actionPtr = createActionToto();
		}

		public void SetTextureActionToto(IntPtr texturePtr)
		{
			setTextureActionToto(_actionPtr, texturePtr);
		}

	}

}
