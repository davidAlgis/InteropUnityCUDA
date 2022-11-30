using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{

    /// <summary>
    /// This class will handle interoperability between Unity/Graphics API/GPGPU Technology
    /// It is used to register and call particular action (other unity plugin
    /// that are using interoperability plugin)
    /// </summary>
    public abstract class InteropHandler : MonoBehaviour
    {
        private const string _dllPluginInterop = "PluginInteropUnityCUDA";

        [DllImport(_dllPluginInterop)]
        protected static extern IntPtr SetTextureFromUnity(IntPtr texture, int w, int h);
        
        [DllImport(_dllPluginInterop)]
        private static extern void StartLog();
        
        
        [DllImport(_dllPluginInterop)]
        private static extern void SetTime(float time);
        
        [DllImport(_dllPluginInterop)]
        private static extern IntPtr GetRenderEventFunc();

        [DllImport(_dllPluginInterop)]
        private static extern int RegisterAction(IntPtr action);
        
        [DllImport(_dllPluginInterop)]
        protected static extern int DoAction(int eventID);
        
        
        [DllImport(_dllPluginInterop)]
        private static extern void InitializeRegisterActions(int reserveCapacity);
        
        

        private readonly Dictionary<int, ActionUnity> _registeredActions = new();
        protected readonly Dictionary<string, int> _actionsNames = new();

        protected virtual int ReserveCapacity => 16;

        protected void Start() 
        {
            StartLog();
            InitializeRegisterActions(ReserveCapacity);
            // GL.IssuePluginEvent(GetRenderEventFunc(), -1);
            //Here create and register your actions
            InitializeActions();
        }

        protected virtual void InitializeActions() { }

        protected void Update()
        {
            SetTime(Time.time);
            CallActions();
        }

        protected virtual void CallActions(){}

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
