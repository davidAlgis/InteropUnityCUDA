using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace Utilities
{
    public class ParticlesDrawer : MonoBehaviour
    {
        [SerializeField] private Material _material;
        private ComputeBuffer _particlesComputeBuffer;
        private int _nbrParticles;
        private float _particlesSizeForRender;
        
        private static readonly int _particlesProp = Shader.PropertyToID("particles");
        private static readonly int _sizeParticles = Shader.PropertyToID("sizeParticles");

        private void Awake()
        {
            if (_material == null)
            {
                Debug.LogError("Set a material in inspector for ParticlesDrawer in " + gameObject.name);
                return;
            }
        }
        

        public void InitParticlesBuffer(ComputeBuffer computeBuffer, int nbrParticles, float particlesSizeForRender)
        {
            _nbrParticles = nbrParticles;
            _particlesComputeBuffer = computeBuffer;
            _particlesSizeForRender = particlesSizeForRender;
            
            var x = new float4[_nbrParticles];
            for (int i = 0; i < _nbrParticles; i++)
                x[i] = new float4(0f,0f,0f, 1f);
            
            _particlesComputeBuffer.SetData(x);
            _material.SetBuffer(_particlesProp, computeBuffer);
        }

        private void OnPostRender()
        {
            var x = new float4[_nbrParticles];
            for (int i = 0; i < _nbrParticles; i++)
                x[i] = new float4(0f,0f,0f, 1f);
            
            _particlesComputeBuffer.SetData(x);;
            
            _material.SetPass(0);
            _material.SetBuffer(_particlesProp, _particlesComputeBuffer);
            _material.SetFloat(_sizeParticles, _particlesSizeForRender);
            Graphics.DrawProceduralNow(MeshTopology.Points, _nbrParticles, 1);
        }

        private void OnDestroy()
        {
            if (_particlesComputeBuffer != null)
            {
                _particlesComputeBuffer.Release();
            }
        }
    }
}