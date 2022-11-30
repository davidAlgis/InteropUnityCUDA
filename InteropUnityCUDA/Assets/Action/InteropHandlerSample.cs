using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ActionUnity
{

    public class InteropHandlerSample : InteropHandler
    {
        
        [SerializeField] private RawImage _rawImage;
        private RenderTexture _rt;
        
        [SerializeField] private int _resolution = 256;
		

        private void CreateTexture()
        {
	        _rt = new RenderTexture(_resolution, _resolution, 0
		        , RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
	        {
		        useMipMap = false,
		        autoGenerateMips = false,
		        anisoLevel = 6,
		        filterMode = FilterMode.Trilinear,
		        wrapMode = TextureWrapMode.Clamp,
		        enableRandomWrite = true
	        };

	        print(_rt.Create());
	        _rawImage.texture = _rt;
        }


        protected override void InitializeActions()
        {
	        base.InitializeActions();
            CreateTexture();
            print(_rt.GetNativeTexturePtr());
            ActionUnitySampleTexture actionUnitySampleTexture = new ActionUnitySampleTexture(_rt);
            RegisterActionUnity(actionUnitySampleTexture, "sample");
            CallActionStart("sample");
        }

        protected override void CallUpdateActions()
        {
	        base.CallUpdateActions();
            CallActionUpdate("sample");
        }
    }

}