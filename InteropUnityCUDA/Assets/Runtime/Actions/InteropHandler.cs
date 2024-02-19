using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
    [Flags]
    public enum ErrorBehavior
    {
        DO_NOTHING = 1,
        ASSERT = 2,
        DISABLE_ACTION = 4
    }


    /// <summary>
    ///     This class will handle interoperability between Unity/Graphics API/GPGPU Technology
    ///     It is used to register and call particular action (other unity plugin
    ///     that are using interoperability plugin)
    ///     Derived from this class if you want to use interoperability
    /// </summary>
    public abstract class InteropHandler : MonoBehaviour
    {
        
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        private const string _dllPluginInterop = "d_PluginInteropUnityCUDA";
#else
        private const string _dllPluginInterop = "PluginInteropUnityCUDA";
#endif
        // dictionary with the string id (more convenient for user) and the corresponding integer id which 
        // is used in PluginInterop
        private readonly Dictionary<string, int> _actionsNames = new();


        // dictionary will all registered action
        private readonly Dictionary<int, ActionUnity> _registeredActions = new();

        protected virtual int ReserveCapacity => 16;

        private void Awake()
        {
            if (IsSupported() == false) Debug.LogError("Interoperability is not supported.");

            SetErrorBehavior(ErrorBehavior.DISABLE_ACTION);
        }


        [DllImport(_dllPluginInterop)]
        private static extern void StartLog();

        [DllImport(_dllPluginInterop)]
        protected static extern int GetErrorBehavior();

        [DllImport(_dllPluginInterop)]
        private static extern void SetErrorBehavior(int behavior);

        [DllImport(_dllPluginInterop)]
        public static extern bool IsSupported();

        [DllImport(_dllPluginInterop)]
        private static extern void SetTimeInterop(float time);

        [DllImport(_dllPluginInterop)]
        private static extern IntPtr GetRenderEventFunc();

        [DllImport(_dllPluginInterop)]
        private static extern int RegisterAction(IntPtr action);

        [DllImport(_dllPluginInterop)]
        private static extern void InitializeRegisterActions(int reserveCapacity);

        public void InitializeInteropHandler()
        {
            // initialize log
            StartLog();

            InitializeRegisterActions(ReserveCapacity);

            //Here create and register your actions
            InitializeActions();
        }

        /// <summary>
        ///     Create and register your actions in this function
        ///     This function is called by MonoBehaviour::Start()
        /// </summary>
        protected virtual void InitializeActions()
        {
        }

        public void UpdateInteropHandler()
        {
            //to update the time used in InteropDll
            SetTimeInterop(Time.time);
            UpdateActions();
        }

        /// <summary>
        ///     Call the update functions of your actions in
        ///     This function is called by MonoBehaviour::Update()
        /// </summary>
        protected virtual void UpdateActions()
        {
        }

        public void OnDestroyInteropHandler()
        {
            OnDestroyActions();
        }

        /// <summary>
        ///     Call the on destroy functions of your actions in
        ///     This function is called by MonoBehaviour::OnDestroy()
        /// </summary>
        protected virtual void OnDestroyActions()
        {
        }

        protected void SetErrorBehavior(ErrorBehavior errorBehavior)
        {
            SetErrorBehavior((int)errorBehavior);
        }

        /// <summary>
        ///     Register an action in PluginInteropUnityCUDA. This is necessary to call an action with CallAction functions.
        /// </summary>
        /// <param name="action">
        ///     ActionUnity used to register the action, the member _actionPtr must be filled
        ///     before using this function.
        /// </param>
        /// <param name="actionName">Id of the action, use this id to call your action</param>
        protected void RegisterActionUnity(ActionUnity action, string actionName)
        {
            if (_actionsNames.ContainsKey(actionName))
            {
                Debug.LogError("Unable to register action with actionName " + actionName +
                               ", because there is already an action with the same name");
                return;
            }

            var key = RegisterAction(action.ActionPtr);
            _actionsNames.Add(actionName, key);
            _registeredActions.Add(key, action);
        }

        /// <summary>
        ///     Get action unity
        /// </summary>
        /// <param name="actionName">name associated to this action</param>
        /// <returns>the action to get</returns>
        protected ActionUnity GetActionUnity(string actionName)
        {
            if (_actionsNames.TryGetValue(actionName, out var key)) return _registeredActions[key];

            Debug.LogError("Unable to find any action with this name : " + actionName);
            return null;
        }


        /// <summary>
        ///     Represent the 3 different function in an Action
        /// </summary>
        private enum ActionType
        {
            Start,
            Update,
            OnDestroy
        }

        #region ACTION_CALLER

        /// <summary>
        ///     Call the function that corresponds to <paramref name="actionType" /> in action register with the name
        ///     <paramref name="actionName" /> by giving the id associated to this action/actionType to the plugin event.
        ///     The action will only be called when the render thread is available.
        ///     <see href="https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html" />
        ///     UNSAFE will not check if action was already registered.
        /// </summary>
        /// <param name="actionName">
        ///     register name of the action. (<c>RegisterActionUnity</c> function for
        ///     more details)
        /// </param>
        /// <param name="actionType">defined which function in action will be called</param>
        private void CallFunctionInAction(string actionName, ActionType actionType)
        {
            // function that work and graphics object (texture, vertex buffer,...) has to be called in render thread
            // therefore, we use this function which will make sure our plugin function is called in render thread
            // the eventId is defined by the action that we are using _actionsNames[actionName] defined the id for 
            // the action and the actionType defined which function in action must be called :
            // 0 -> Start
            // 1 -> Update
            // 2 -> OnDestroy
            GL.IssuePluginEvent(GetRenderEventFunc(), 3 * _actionsNames[actionName] + (int)actionType);
        }

        /// <summary>
        ///     Call the Start function in action register with the name <paramref name="actionName" />
        ///     see <c>CallFunctionInAction</c> function for more details.
        ///     UNSAFE will not check if action was already registered.
        /// </summary>
        /// <param name="actionName">
        ///     register name of the action. (see <c>RegisterActionUnity</c> function for
        ///     more details)
        /// </param>
        protected void CallFunctionStartInAction(string actionName)
        {
            CallFunctionInAction(actionName, ActionType.Start);
        }

        /// <summary>
        ///     Call the Update function in action register with the name <paramref name="actionName" />
        ///     see <c>CallFunctionInAction</c> function for more details.
        ///     UNSAFE will not check if action was already registered.
        /// </summary>
        /// <param name="actionName">
        ///     register name of the action. (see <c>RegisterActionUnity</c> function for
        ///     more details)
        /// </param>
        protected void CallFunctionUpdateInAction(string actionName)
        {
            CallFunctionInAction(actionName, ActionType.Update);
        }

        /// <summary>
        ///     Call the Destroy function in action register with the name <paramref name="actionName" />
        ///     see <c>CallFunctionInAction</c> function for more details.
        ///     UNSAFE will not check if action was already registered.
        /// </summary>
        /// <param name="actionName">
        ///     register name of the action. (see <c>RegisterActionUnity</c> function for
        ///     more details)
        /// </param>
        protected void CallFunctionOnDestroyInAction(string actionName)
        {
            CallFunctionInAction(actionName, ActionType.OnDestroy);
        }

        #endregion
    }
}