using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{

	public class ActionUnitySampleVertexBuffer : ActionUnity
	{
		const string _dllSampleBasic = "SampleBasic";

		[DllImport(_dllSampleBasic)]
		private static extern IntPtr createActionSampleVertexBufferBasic(IntPtr vertexBufferPtr, int size);
		
		public ActionUnitySampleVertexBuffer(ComputeBuffer computeBuffer, int size) : base()
		{
			_actionPtr = createActionSampleVertexBufferBasic(computeBuffer.GetNativeBufferPtr(), size);
		}
	}

}
