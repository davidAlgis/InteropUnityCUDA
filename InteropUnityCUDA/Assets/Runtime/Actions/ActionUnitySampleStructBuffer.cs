using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
    public class ActionUnitySampleStructBuffer : ActionUnity
    {
        private const string _dllSampleBasic = "SampleBasic";

        public ActionUnitySampleStructBuffer(ComputeBuffer computeBuffer, int size)
        {
            _actionPtr = createActionSampleStructBufferBasic(computeBuffer.GetNativeBufferPtr(), size);
        }

        [DllImport(_dllSampleBasic)]
        private static extern IntPtr createActionSampleStructBufferBasic(IntPtr structBufferPtr, int size);
    }
}