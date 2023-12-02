#pragma once
#include <sutil/vec_math.h>

namespace Trace {

    class Camera {

    public:
        


    private:
        float3 _position = make_float3(0.0f, 0.0f, 0.0f);
        float3 _target = make_float3(0.0f);
        float vFOV = 0.0f;
        float AR = 0.0f;

        float3 u = make_float3(1.0f, 0.0f, 0.0f);
        float3 v = make_float3(0.0f, 1.0f, 0.0f);
        float3 w = make_float3(0.0f, 0.0f, 1.0f);

    public:

        __host__ Camera() {}
        __host__ Camera(const float3& position, const float3& target, const float vFOV, const float AR) :
            _position(position), _target(target), vFOV(vFOV), AR(AR) {

            updateRotationMatrix();
        }

        __host__ __device__ float3 getPosition() { return _position; }
        __host__ __device__ float3 getTarget() { return _target; }

        __host__ void updateRotationMatrix() {
            sensorFrame(getPosition(), getTarget());
            updateViewPort();
        }

        __host__ void setPosition(float3& position) {
            this->_position = position;
            updateRotationMatrix();
        }
        __host__ void setTarget(float3& target) {
            this->_target = target;
            updateRotationMatrix();
        }

        __host__ float computeDistanceToTarget() {

            return length(_target - _position);
        }


        

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

        __host__ void updateViewPort() {
            float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
            float halfWidth = AR * halfHeight;
            u *= halfWidth;
            v *= halfHeight;
        }

    };

};