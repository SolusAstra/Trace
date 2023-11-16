#pragma once
#include <sutil/vec_math.h>

namespace Trace {

    class Ray {

    public:

        float3 org;
        float3 dir;
        float3 albedo;
        float t = 0.0f;

        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const float3& origin, const float3& direction) {
            this->org = origin;
            this->dir = direction;
        }
        __device__ void set(const Ray& ray) {
            this->org = ray.org;
            this->dir = ray.dir;
            this->t = ray.t;
        }

        __forceinline __device__ void cast() {
            org += t * dir;
        }

        __forceinline __device__ float3 getCast() const {
            return org + t * dir;
        }

        __forceinline __device__ void cast(float _t) {
            org += _t * dir;
        }

        __forceinline __device__ void castFrom(const Ray& a) {
            org = a.org + t * a.dir;
        }

        __forceinline __device__ void computeNormal(const float3& ref) {
            dir = normalize(org - ref);
        }

        __forceinline __device__ void correctFloatingPointBias() {
            org += dir * 0.0001f;
        }

    };
};
