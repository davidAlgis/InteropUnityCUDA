using System;
using UnityEngine;

namespace ActionUnity
{
    /// <summary>
    /// This class is an abstract class to store Action class (the one from the native plugin)
    /// You need to derived from this class and get a pointer to the action class by adding an appropriate
    /// function in native plugin.
    /// </summary>
    public abstract class ActionUnity
    {
        protected IntPtr _actionPtr;
        
        public IntPtr ActionPtr => _actionPtr;

        protected ActionUnity()
        {
        }
    }

}
