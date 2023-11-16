#pragma once
#include <sutil/vec_math.h>

namespace Trace {

    class Camera {

    public:
        float3 position = make_float3(0.0f, 0.0f, 0.0f);
        float3 u = make_float3(1.0f, 0.0f, 0.0f);
        float3 v = make_float3(0.0f, 1.0f, 0.0f);
        float3 w = make_float3(0.0f, 0.0f, 1.0f);

    public:

        __host__ Camera() {}
        __host__ Camera(const float3& position, const float3& target, const float vFOV, const float AR);
        __host__ void update(const float3& position, const float3& target, const float vFOV, const float AR);

        __forceinline __device__ float3 getPixelPosition(float i, float j) 
        {
            return u * (i - 0.5f) + v * (j - 0.5f) - w;
        }

    private:

        __host__ void sensorFrame(const float3& position, const float3& target)
        {
            const float3 n = make_float3(0.0f, 1.0f, 0.0f);

            // Calculate the camera frame
            w = normalize(position - target);
            u = normalize(cross(n, w));
            v = normalize(cross(w, u));
        }

    };

    __host__ inline Camera::Camera(const float3& position, const float3& target, const float vFOV, const float AR)
    {
        this->position = position;

        // Calculate the camera frame
        sensorFrame(position, target);

        // Viewport Calculations
        float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
        float halfWidth = AR * halfHeight;
        u *= halfWidth;
        v *= halfHeight;
    }

    __host__ inline void Camera::update(const float3& position, const float3& target, const float vFOV, const float AR)
    {
        this->position = position;

        // Calculate the camera frame
        sensorFrame(position, target);

        // Viewport Calculations
        float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
        float halfWidth = AR * halfHeight;
        u *= halfWidth;
        v *= halfHeight;
    }

};