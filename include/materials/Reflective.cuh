#pragma once
#include "Material.cuh"

namespace Trace {

    class Reflective : public Material {

    public:
        float _roughness = 0.0f;

    public:

        __forceinline __host__ __device__ Reflective(float3 albedo, float roughness) {
            this->_albedo = albedo;
            this->_roughness = roughness;
        }

        __forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const override;
        __forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const override;

    };

    inline bool Reflective::scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const
    {
        // Set attributes of outgoing ray
        ray.org = hit.point;
        ray.albedo = (_albedo);

        if (_roughness > 0.0f && rand) {
            ray.dir = (normalize(reflect(ray.dir, hit.normal) + _roughness * randFloat3(rand)));
        }
        else {
            ray.dir = (reflect(ray.dir, hit.normal));
        }

        return true;
    }

    inline bool Reflective::emitted(Trace::Ray& ray, Trace::Record& hit) const 
    { 
        return false; 
    }
};