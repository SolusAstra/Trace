#pragma once
#include "utils/Ray.h"
#include "utils/Record.h"

namespace Trace {

    namespace DataType {
        struct SoA;
        struct AoS;
    };

    class Primitive {

    public:

        __host__ __device__ Primitive() {}

        __forceinline __device__ virtual bool hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) const = 0;

        __forceinline __device__ void updatePayload(const Trace::Ray& ray, Trace::Record& hit, float root) const {

            hit.t = root;
            hit.point = ray.org + hit.t * ray.dir;

            hit.normal = computeNormal(hit.point);

            // Offset hit point to avoid floating point bias
            hit.point += hit.normal * 0.0001f;
            if (dot(hit.normal, ray.dir) > 0.0f) {
                hit.normal = -hit.normal;
            }

            hit.material = getMaterial();

        }


        __forceinline __device__ virtual Trace::Material* getMaterial() const = 0;
        __forceinline __device__ virtual float3 computeNormal(float3& point) const = 0;
    };
};