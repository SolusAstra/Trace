#pragma once
#include "Material.cuh"

namespace Trace {

    class Light : public Material {

    public:
        float3 _albedo = make_float3(0.0f);

    public:

        __forceinline __host__ __device__ Light(float3 albedo) : _albedo(albedo) {}

        __forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const override;
        __forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const override;
    
    };


    inline bool Light::scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const 
    {
        return false;
    }
    inline bool Light::emitted(Trace::Ray& ray, Trace::Record& hit) const 
    {
        ray.albedo = (_albedo);
        return true;
    }
};