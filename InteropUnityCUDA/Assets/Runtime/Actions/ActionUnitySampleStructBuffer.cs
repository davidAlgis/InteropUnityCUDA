using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
    public class ActionUnitySampleStructBuffer : ActionUnity
    {

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        private const string _dllSampleBasic = "d_SampleBasic";
#else
        private const string _dllSampleBasic = "SampleBasic;
#endif
        public ActionUnitySampleStructBuffer(ComputeBuffer computeBuffer, int size)
        {
            _actionPtr = createActionSampleStructBufferBasic(computeBuffer.GetNativeBufferPtr(), size);
        }

        [DllImport(_dllSampleBasic)]
        private static extern IntPtr createActionSampleStructBufferBasic(IntPtr structBufferPtr, int size);
    }
}