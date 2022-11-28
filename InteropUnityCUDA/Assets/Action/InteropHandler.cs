using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.VisualScripting;
using UnityEngine;

namespace ActionUnity
{

    /// <summary>
    /// This class will handle interoperability between Unity/Graphics API/GPGPU Technology
    /// It is used to send graphics type (buffer, texture,...) to interoperability plugin
    /// Moreover, it's used to register and call particular action (other unity plugin
    /// that are using interoperability plugin)
    /// </summary>
    public class InteropHandler : MonoBehaviour
    {
        const string _dllPluginInterop = "PluginInteropUnityCUDA";

        [DllImport(_dllPluginInterop)]
        protected static extern void SetBufferFromUnity(IntPtr buffer, int size);

        [DllImport(_dllPluginInterop)]
        protected static extern void SetTextureFromUnity(IntPtr texture, int w, int h);

        [DllImport(_dllPluginInterop)]
        protected static extern IntPtr GetRenderEventFunc();

        [DllImport(_dllPluginInterop)]
        protected static extern int RegisterAction(IntPtr action);

        private Dictionary<int, ActionUnity> _registeredActions = new Dictionary<int, ActionUnity>();
        private Dictionary<string, int> _actionsNames = new Dictionary<string, int>();

        // Start is called before the first frame update
        void Awake()
        {
            //Here create and register your actions
            ActionUnitySample actionUnitySample = new ActionUnitySample();
            int key = RegisterActionUnity(actionUnitySample, "sample");
        }
        
        private void Update()
        {
            //Here call your actions
            StartCoroutine(CallActionAtEndOfFrames("sample"));
        }

        protected IEnumerator CallActionAtEndOfFrames(string actionName)
        {
            yield return new WaitForEndOfFrame();

            GL.IssuePluginEvent(GetRenderEventFunc(), _actionsNames[actionName]);
        }
        
        protected IEnumerator CallEachRegisteredActionAtEndOfFrames()
        {
            yield return new WaitForEndOfFrame();

            foreach (int index in _registeredActions.Keys)
            {
                GL.IssuePluginEvent(GetRenderEventFunc(), index);
            }
        }
        protected int RegisterActionUnity(ActionUnity action, string actionName)
        {
            if (_actionsNames.ContainsKey(actionName))
            {
                Debug.LogError("Unable to register action with actionName " + actionName +
                               ", because there is already an action with the same name");
                return -1;
            }

            int key = RegisterAction(action.ActionPtr);
            _actionsNames.Add(actionName, key);
            action.InitializeKey(key);
            _registeredActions.Add(key, action);
            return key;
        }
    }

}
