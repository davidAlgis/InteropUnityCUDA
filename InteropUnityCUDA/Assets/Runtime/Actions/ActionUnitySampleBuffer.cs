using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{

	public class ActionUnitySampleVertexBuffer : ActionUnity
	{
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        private const string _dllSampleBasic = "d_SampleBasic";
#else
        private const string _dllSampleBasic = "SampleBasic";
#endif

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionSampleVertexBufferBasic(IntPtr vertexBufferPtr, int size);
		
		public ActionUnitySampleVertexBuffer(ComputeBuffer computeBuffer, int size) : base()
		{
			_actionPtr = createActionSampleVertexBufferBasic(computeBuffer.GetNativeBufferPtr(), size);
		}
	}

}
