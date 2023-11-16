#pragma once
#include "Record.cuh"
#include "Ray.cuh"

namespace Trace {
    
    class SphereSoA {

    public:
        float3* center;
        float* radius;
        int size = 0;

    public:

        __forceinline __host__ __device__ SphereSoA() {}

        __forceinline __device__ float3 normalAt(const int idx, const float3& point) const {
            return normalize(point - center[idx]);
        }

        __forceinline  __device__ bool hitRoot(const int idx, const Trace::Ray& ray, float& root, float t_max) const;

        __forceinline __device__ static void get_sphere_uv(const float3& p, float& u, float& v) {
            // p: a given point on the sphere of radius one, centered at the origin.
            // u: returned value [0,1] of angle around the Y axis from X=-1.
            // v: returned value [0,1] of angle from Y=-1 to Y=+1.
            //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
            //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
            //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

            float pi = 3.141592;
            float theta = acos(-p.y);
            float phi = atan2(-p.z, p.x) + pi;

            u = phi / (2 * pi);
            v = theta / pi;
        }

    };

    __forceinline __device__ bool SphereSoA::hitRoot(const int idx, const Trace::Ray& ray, float& root, float t_max) const {

        float3 SR = ray.org - center[idx];

        // Quadratic coefficients
        float a = dot(ray.dir, ray.dir);
        float b = dot(SR, ray.dir);
        float c = dot(SR, SR) - radius[idx] * radius[idx];
        float D = b * b - a * c;

        // Compute hit
        if (D <= 0.0f) {
            return false;
        }

        float r1 = (-b - sqrtf(D)) / a;
        if (r1 < t_max && r1 > 0.0001f) {
            root = r1;
            return true;
        }

        float r2 = (-b + sqrtf(D)) / a;
        if (r2 < t_max && r2 > 0.0001f) {
            root = r2;
            return true;
        }

        return false;
    }

};