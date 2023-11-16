#pragma once
#include <curand_kernel.h>

#include "utils/Ray.h"
#include "utils/helper.cuh"

namespace Trace {

    class Material {

    public:
        float3 _albedo = make_float3(0.0f);

    public:

        __forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) const = 0;
        __forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const = 0;

    };

};