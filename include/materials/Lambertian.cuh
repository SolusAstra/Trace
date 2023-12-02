#pragma once
#include "Material.cuh"

namespace Trace {

    class Lambertian : public Material {

    public:

        __forceinline __host__ __device__ Lambertian(float3 albedo)  {
            this->_albedo = albedo;
        }

        __forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* local_rand_state = nullptr) const override;
        __forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const override;

    };

    inline bool Lambertian::scatter(Trace::Ray& ray, Trace::Record& hit, curandState* local_rand_state) const 
    {

        // Set attributes of outgoing ray
        ray.org = hit.point;
        ray.albedo = (_albedo);
        ray.dir = (hit.normal);

        // Lambertian scattering
        if (local_rand_state) {
            float3 scatterDirection = normalize(hit.normal + randFloat3(local_rand_state));
            if (dot(scatterDirection, hit.normal) > 0.0f) { ray.dir = (scatterDirection); }
        }

        return true;
    }

    inline bool Lambertian::emitted(Trace::Ray& ray, Trace::Record& hit) const 
    { 
        return false; 
    }
};