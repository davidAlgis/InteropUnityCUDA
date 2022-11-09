//Code written by Przemyslaw Zaworski
//https://github.com/przemyslawzaworski

Shader "Draw Particles"
{
    Properties
    {
        colorParticles("Color", Color) = (1.0,0.0,0.0,1.0)
    }
    SubShader
    {
        Pass
        {
            ZTest Always 
        	Cull off
            CGPROGRAM         
            #pragma vertex vert
            #pragma fragment pixelShader
			#pragma geometry geom
            #pragma target 5.0

            
            float4 colorParticles;
            StructuredBuffer<float4> particles;
            float sizeParticles;
            
            struct vertexIn {
				float4 pos : POSITION;
				float4 color : COLOR;
			};

			struct geomOut {
				float4 pos : POSITION;
				float4 color : COLOR0;
			};

            vertexIn vert(uint id : SV_VertexID)
            {
                vertexIn vs;
                vs.pos = UnityObjectToClipPos(particles[id]);
            	
            	vs.color = colorParticles;
				
            	
            	return vs;
            }

            float4 pixelShader(vertexIn ps) : SV_TARGET
            {
            	return colorParticles;
            }

			[maxvertexcount(4)]
			void geom ( point vertexIn IN[1] , inout TriangleStream<geomOut> triStream )
			{
				vertexIn vertex = IN[0];
				const float2 points[4] = { float2(1,-1) , float2(1,1) , float2(-1,-1) , float2(-1,1) };
				float2 pmul = float2( sizeParticles*(_ScreenParams.y / _ScreenParams.x) , sizeParticles ) * 0.5;
				
				geomOut newVertex;
            	newVertex.color = vertex.color;
				for( int i=0 ; i<4 ; i++ )
				{
					newVertex.pos = vertex.pos + float4( points[i]*pmul , 0 , 0 );
					triStream.Append( newVertex );
				}
			}

            
            ENDCG
        }
    }
}