#pragma once
#include "Ray.cuh"

namespace Trace {



    class Camera {

    public:
        float3 position = make_float3(0.0f, 0.0f, 0.0f);

        float3 u = make_float3(1.0f, 0.0f, 0.0f);
        float3 v = make_float3(0.0f, 1.0f, 0.0f);
        float3 w = make_float3(0.0f, 0.0f, 1.0f);
        //float3x3 Frame = float3x3::identity();

    public:

        __host__ __device__ void sensorFrame(const float3& position, const float3& target) {

            const float3 n = make_float3(0.0f, 1.0f, 0.0f);
            
            // Calculate the camera frame
            w = normalize(position - target);
            u = normalize(cross(n, w));
            v = normalize(cross(w, u));

            //return Frame(u, v, w);
        }

        __host__ Camera() {}
        __host__ Camera(const float3& position, const float3& target, const float vFOV, const float AR);
        __host__ void update(const float3& position, const float3& target, const float vFOV, const float AR);


        // Calculate the position of a given pixel
        __host__ __device__ float3 getPixelPosition(float i, float j) {
            return u * (i - 0.5f) + v * (j - 0.5f) - w;
        }

    };


};