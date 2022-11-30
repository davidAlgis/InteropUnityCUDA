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
        private static extern void StartLog();
        
        [DllImport(_dllPluginInterop)]
        private static extern void SetTime(float time);
        
        [DllImport(_dllPluginInterop)]
        private static extern IntPtr GetRenderEventFunc();

        [DllImport(_dllPluginInterop)]
        private static extern int RegisterAction(IntPtr action);
        
        
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
            CallUpdateActions();
        }

        protected virtual void CallUpdateActions(){}
        
        
        protected void CallActionByName(string actionName, ActionType actionType)
        {
            GL.IssuePluginEvent(GetRenderEventFunc(), 3*_actionsNames[actionName] + (int)actionType);
        }
        
        protected void CallActionStart(string actionName)
        {
            CallActionByName(actionName,ActionType.Start);
        }
        
        protected void CallActionUpdate(string actionName)
        {
            CallActionByName(actionName,ActionType.Update);
        }
        
        protected void CallActionOnDestroy(string actionName)
        {
            CallActionByName(actionName,ActionType.OnDestroy);
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
        
        
        
        protected enum ActionType
        {
            Start,
            Update,
            OnDestroy
        }
    }
}
