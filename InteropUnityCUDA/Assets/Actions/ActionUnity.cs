using System;
using UnityEngine;

namespace ActionUnity
{
    public abstract class ActionUnity
    {
        private int _key = -1;
        protected IntPtr _actionPtr;
        
        public IntPtr ActionPtr => _actionPtr;

        protected ActionUnity()
        {
        }

        public void InitializeKey(int key)
        {
            if (_key != -1)
            {
                Debug.LogError("Cannot edit a key that has already been initialized");
                return;
            }

            _key = key;
        }
    }

}
