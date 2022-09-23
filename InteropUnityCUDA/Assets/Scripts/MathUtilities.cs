using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Mathematics;
using Random = UnityEngine.Random;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine.Assertions;

namespace ArcBlanc.Utilities
{



    public class MathUtilities
    {
        /// <summary>
        /// Split a given float3 list to three list of his components
        /// </summary>
        /// <param name="origin">list of origin</param>
        /// <param name="xComp">list of the x component</param>
        /// <param name="yComp">list of the y component</param>
        /// <param name="zComp">list of the z component</param>
        public static void ListVectorReduction(List<float3> origin, out List<float> xComp, out List<float> yComp,
            out List<float> zComp)
        {
            xComp = new List<float>();
            yComp = new List<float>();
            zComp = new List<float>();
            
            if (origin != null)
            {
                foreach (var v in origin)
                {
                    xComp.Add(v.x);
                    yComp.Add(v.y);
                    zComp.Add(v.z);
                }
            }
        }

        /// <summary>
        /// Give the list of the norm of a given list of vector
        /// </summary>
        /// <param name="origin">list of origin</param>
        /// <param name="norm">list of norm</param>
        public static void ListVectorToListNorm(List<float3> origin, out List<float> norm)
        {
            norm = new List<float>();
            
            if (origin != null)
            {
                foreach (var v in origin)
                {
                    norm.Add(math.length(v));
                }
            }
        }
        
        /// <summary>
        /// Get an entire column of a matrix
        /// </summary>
        /// <param name="matrix">Input matrix</param>
        /// <param name="columnNumber">the number of the column</param>
        /// <typeparam name="T">type of matrix</typeparam>
        /// <returns>The column that has been retrieve from matrix</returns>
        public static T[] GetColumn<T>(T[,] matrix, int columnNumber)
        {
            return Enumerable.Range(0, matrix.GetLength(0))
                .Select(x => matrix[x, columnNumber])
                .ToArray();
        }
        
        public static float NumberReduction(float numberToReduce, float maxOrder)
        {
            float logarithmicNumber = math.max(2.0f, math.log2(maxOrder) / 2f);
            
            return (float)System.Math.Log((double)2 * numberToReduce + 1, (double)logarithmicNumber);
        }

        public static float3 VectorReduction(float3 vectorToReduce, float maxOrder)
        {
            if (math.length(vectorToReduce) < 1e-3)
                return vectorToReduce;
            float sizeReduction = NumberReduction(math.length(vectorToReduce), math.abs(maxOrder));
            return math.normalize(vectorToReduce) * sizeReduction;
        }
        

        /// <summary>
        /// Return true if x is a power of 2
        /// </summary>
        public static bool IsPowerOfTwo(int x) 
        {
            return (x & (x - 1)) == 0;
        }

        /// <summary>
        /// Return the nearest (lower) power of two of x
        /// </summary>
        public static int CeilPowerOfTwo(int x)
        {
            if (x < 2)
                return 1;
            
            return (int)Math.Pow(2, (int)Math.Log(x - 1, 2) + 1);
        }

        /// <summary>
        /// truncate the number of decimal of a given float
        /// eg : 13.654161 with numberOfDecimal = 3 will be transformed to 13.65
        /// </summary>
        /// <param name="numberToCut">float to truncate</param>
        /// <param name="numberOfDecimal">number of decimal to have after truncation</param>
        public static void TruncateNumber(ref float numberToCut, int numberOfDecimal)
        {
            float tenPowerNumberDecimal = math.pow(10f, numberOfDecimal);
            numberToCut = math.round(numberToCut * tenPowerNumberDecimal) / tenPowerNumberDecimal;
        }
        
        /// <summary>
        /// truncate the number of decimal of a given float3
        /// </summary>
        /// <param name="vectorToCut">float3 to truncate</param>
        /// <param name="numberOfDecimal">number of decimal to have after truncation</param>
        public static void TruncateNumber(ref float3 vectorToCut, int numberOfDecimal)
        {
            TruncateNumber(ref vectorToCut.x, numberOfDecimal);
            TruncateNumber(ref vectorToCut.y, numberOfDecimal);
            TruncateNumber(ref vectorToCut.z, numberOfDecimal);
        }

        public static float NormalRandom()
        {
            return math.cos(2 * math.PI * Random.value) * math.sqrt(-2 * math.log(UnityEngine.Random.value));
        }
        
        
        /// <summary>
        /// Get the local to world matrix only from pos, eulerAngle and scale
        /// https://answers.unity.com/questions/1656903/how-to-manually-calculate-localtoworldmatrixworldt.html
        /// </summary>
        /// <returns>A 4x4 matrix which transform a point from local to world</returns>
        public static  Matrix4x4 GetLocalToWorldMatrix(Vector3 pos, Vector3 eulerAngle, Vector3 scale)
        {
            return Matrix4x4.TRS(pos,  Quaternion.Euler(eulerAngle), scale);
        }
        
        
        /// <summary>
        /// Interpolate from the range [-1,1] to [-100,100].
        /// </summary>
        public static float OneBaseToHundredBase(float value)
        {

            return Mathf.Clamp(value * 100f, -100, 100);
        }

        
        
        #region ConvertInertia
        
        /// <summary>
        /// Use to convert a inertia matrix 3x3 to inertia tensor and inertia tensor rotation used in rigidbody component. 
        /// This function is based on the code of PhysX function PxDiagonalize : 
        /// https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/master/PxShared/src/foundation/src/PsMathUtils.cpp
        /// moreover : https://answers.unity.com/questions/1484654/how-to-calculate-inertia-tensor-and-tensor-rotatio.html
        /// </summary>
        /// <param name="inertiaTensorMatrix">the input a matrix which represent the inertia matrix</param>
        /// <param name="inertiaTensorRotation">the inertia tensor of rotation</param>
        /// <returns></returns>
        public static Vector3 InertiaTensorMatToInertiaVec(float3x3 inertiaTensorMatrix, ref Quaternion inertiaTensorRotation)
        {
            // jacobi rotation using quaternions (from an idea of Stan Melax, with fix for precision issues)
            const int MAX_ITERS = 24;

            Quaternion q = Quaternion.identity; // PxQuat(PxIdentity);

            float3x3 d = new float3x3();
            for(int i = 0; i<MAX_ITERS; i++)
            {
                float3x3 axes = QuaternionToMatrix3x3(q);
                d = math.mul(math.mul(math.transpose(axes), inertiaTensorMatrix), axes);
 
                float d0 = math.abs(d[1][2]), d1 = math.abs(d[0][2]), d2 = math.abs(d[0][1]);
                int a = d0 > d1 && d0 > d2 ? 0 : d1 > d2 ? 1 : 2; // rotation axis index, from largest off-diagonal
                // element

                int a1 = (a + 1) % 3; 
                int a2 = (a1 + 1) % 3;
                if(d[a1][a2] == 0.0f || math.abs(d[a1][a1] - d[a2][a2]) > 2e6f * math.abs(2.0f * d[a1][a2]))
                    break;
 
                float w = (d[a1][a1] - d[a2][a2]) / (2.0f * d[a1][a2]); // cot(2 * phi), where phi is the rotation angle
                float absw = math.abs(w);

                Quaternion r;
                if(absw > 1000)
                    r = IndexedRotation(a, 1 / (4 * w), 1f); // h will be very close to 1, so use small angle approx instead
                else
                {
                    float t = 1 / (absw + math.sqrt(w * w + 1)); // absolute value of tan phi
                    float h = 1 / math.sqrt(t * t + 1);          // absolute value of cos phi

                    r = IndexedRotation(a, math.sqrt((1 - h) / 2) * math.sign(w), math.sqrt((1 + h) / 2));
                }
                q = Quaternion.Normalize(q * r);
            }

            inertiaTensorRotation = q;
            return new Vector3(d[0][0], d[1][1], d[2][2]);
        }
        
        public static float3x3 QuaternionToMatrix3x3(Quaternion q)
        {
            float x = q.x;
            float y = q.y;
            float z = q.z;
            float w = q.w;

            float x2 = x + x;
            float y2 = y + y;
            float z2 = z + z;

            float xx = x2 * x;
            float yy = y2 * y;
            float zz = z2 * z;

            float xy = x2 * y;
            float xz = x2 * z;
            float xw = x2 * w;

            float yz = y2 * z;
            float yw = y2 * w;
            float zw = z2 * w;

            return new float3x3(1.0f - yy - zz, xy - zw, xz + yw, xy + zw, 1.0f - xx - zz, yz - xw, xz - yw, yz + xw, 1.0f - xx - yy);
        }

        public static Quaternion IndexedRotation(int axis, float s, float c)
        {
            float[] v = new float[3] { 0, 0, 0 };
            v[axis] = s;
            return new Quaternion(v[0], v[1], v[2], c);
        }

        #endregion


        public static float Mean(List<float> list)
        {
            if (list == null)
            {
                UnityEngine.Debug.LogWarning("Unable to get mean on a null list");
                return 0.0f;
            }

            float mean = 0f;
            foreach (var x in list)
            {
                mean += x;
            }

            mean /= list.Count;
            
            return mean;
        }
        
        public static double Mean(List<double> list)
        {
            if (list == null)
            {
                UnityEngine.Debug.LogWarning("Unable to get mean on a null list");
                return 0.0f;
            }

            double mean = 0f;
            foreach (var x in list)
            {
                mean += x;
            }

            mean /= list.Count;
            
            return mean;
        }
        
        public static double Variance(List<double> list, double mean)
        {
            if (list == null)
            {
                UnityEngine.Debug.LogWarning("Unable to get mean on a null list");
                return 0.0f;
            }

            double variance = 0;
            foreach (var x in list)
            {
                variance += x*x;
            }

            variance /= list.Count;

            variance -= mean * mean;
            return variance;
        }
        
        


        public static void Array2DToArray1D<T>(T[,] array2D, ref T[] array1D) where T : struct
        {
            int len1D = array1D.Length;
            int width = array2D.GetLength(0);
            int height = array2D.GetLength(1);
            int len2D =  width * height;
            if (len1D < len2D)
            {
                Debug.LogError("Unable to convert 2d array to 1d array, because the 1d array has less element than the 2d array");
                return;
            }

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    array1D[i * height + j] = array2D[i, j];
                }
            }
            // Buffer.BlockCopy(array2D, 0,array1D, 0, len2D * Marshal.SizeOf(default(T)));
        }
        
        public static void Array1DToArray2D<T>(T[] array1D, ref T[,] array2D) where T : struct
        {
            int len1D = array1D.Length;
            int width = array2D.GetLength(0);
            int height = array2D.GetLength(1);
            int len2D =  width * height;
            if (len2D < len1D)
            {
                Debug.LogError("Unable to convert 1d array to 2d array, because the 2d array has less element than the 1d array");
                return;
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    array2D[i,j] = array1D[i * height + j];
                }
            }
        }
        
        /// <summary>
        /// return true if one of the component is NaN
        /// </summary>
        /// <param name="v">Vector to test if is nan</param>
        /// <returns>true if one of the component is NaN, else false</returns>
        public static bool IsNaN(Vector3 v)
        {
            return float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z);
        }

        /// <summary>
        /// return true if one of the component is Infinity
        /// </summary>
        /// <param name="v">Vector to test if is Infinity</param>
        /// <returns>true if one of the component is Infinity, else false</returns>
        public static bool IsInfinity(Vector3 v)
        {
            return float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z);
        }

        /// <summary>s
        /// return true if one of the component is Infinity or NaN
        /// </summary>
        /// <param name="v">Vector to test if is Infinity or NaN</param>
        /// <returns>true if one of the component is Infinity or NaN, else false</returns>
        public static bool IsNaNOrInfinity(Vector3 v)
        {
            return IsNaN(v) || IsInfinity(v);
        }


        /// <summary>
        /// Integral approximation  with the trapezoidal rules
        /// of the function f(x) defined by 2 array x and fx, where f(x[i]) = fx[i]
        /// x and fx must be well ordered and of the same size.
        /// </summary>
        /// <param name="x">Antecedent regards to the function f</param>
        /// <param name="fx">image of x regards to the function f</param>
        /// <param name="errorOrder">Order of error using the trapezoidal rules</param>
        /// <returns>Integral approximation of function f</returns>
        public static float IntegralApproximation(float[] x, float[] fx, out float errorOrder)
        {
            if (x.Length != fx.Length)
            {
                Debug.LogWarning("Unable to get integral if x and y are not of the same size");
                errorOrder = 0f;
                return 0f;
            }

            int n = x.Length - 1;
            
            float integral = 0;

            for (int i = 0; i < n; ++i)
            {
                integral += (x[i+1] - x[i])/2f*(fx[i+1] + fx[i]);
            }

            errorOrder = math.pow(x[n] - x[0], 3f) / (12f * n * n);
            return integral;
        }
        
        
        public static unsafe void MemCpy <TSrc,TDst> ( TSrc[] src , TDst[] dst )
            where TSrc : struct
            where TDst : struct
        {
            int srcSize = src.Length * UnsafeUtility.SizeOf<TSrc>();
            int dstSize = dst.Length * UnsafeUtility.SizeOf<TDst>();
            Assert.AreEqual( srcSize , dstSize , $"{nameof(srcSize)}:{srcSize} and {nameof(dstSize)}:{dstSize} must be equal." );
            void* srcPtr = UnsafeUtility.PinGCArrayAndGetDataAddress( src , out ulong srcHandle );
            void* dstPtr = UnsafeUtility.PinGCArrayAndGetDataAddress( dst , out ulong dstHandle );
            UnsafeUtility.MemCpy( destination:dstPtr , source:srcPtr , size:srcSize );
            UnsafeUtility.ReleaseGCObject( srcHandle );
            UnsafeUtility.ReleaseGCObject( dstHandle );
        }

        
        [BurstCompile(FloatPrecision.Low, FloatMode.Fast, CompileSynchronously = true)]
        public struct TransformToGlobal : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> InputPosLocal;
            [ReadOnly] public NativeArray<float4x4> MatrixTransform;
            [WriteOnly] public NativeArray<float3> OutputPosGlobal;
            
            

            public void Execute(int index)
            {
                float4 posLocal = new float4(InputPosLocal[index], 1f);
                OutputPosGlobal[index] = math.mul(MatrixTransform[index], posLocal).xyz;
            }
        }
        

        public class Smoother
        {
            private float _oldValue;
            private readonly float _maxAbsDerivative;
            
            
            public Smoother(float y, float maxAbsDerivative = 10f)
            {
                _oldValue = y;
                _maxAbsDerivative = maxAbsDerivative;
            } 
            
            
            /// <summary>
            /// Will smooth the value by "clamping" his derivatives 
            /// </summary>
            /// <param name="dt">delta time between this value and the las</param>
            /// <param name="y">new value</param>
            /// <returns>values that has been smoothed</returns>
            public void Smooth(float dt, ref float y)
            {
                float epsilon = dt;
                
                if (math.abs(epsilon) < 1e-5)
                {
                    Debug.LogWarning("Can't smooth value on the same times");
                    
                    _oldValue = y;
                    return;
                }

                float dy = (y - _oldValue) / epsilon;

                if (dy > _maxAbsDerivative)
                {
                    y = _maxAbsDerivative * epsilon + _oldValue;
                }
                
                if (dy < -_maxAbsDerivative)
                {
                    y = -_maxAbsDerivative * epsilon + _oldValue;
                }

                _oldValue = y;
            }
            
        }
        
            
    }
    
    public class MovingAverage
    {
        private readonly int _k;
        private readonly long[] _values;

        private int _index = 0;
        private long _sum = 0;

        public MovingAverage(int k)
        {
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Must be greater than 0");

            _k = k;
            _values = new long[k];
        }

        public long Update(long nextInput)
        {
            // calculate the new sum
            _sum = _sum - _values[_index] + nextInput;

            // overwrite the old value with the new one
            _values[_index] = nextInput;

            // increment the index (wrapping back to 0)
            _index = (_index + 1) % _k;

            // calculate the average
            return _sum / _k;
        }
    }
}

