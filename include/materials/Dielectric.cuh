#pragma once
#include "Material.cuh"

namespace Trace {

    class Dielectric : public Material {

    private:

        float _refractionIndex;

    public:

        __forceinline __device__ Dielectric(float refractionIndex) : _refractionIndex(refractionIndex) {};

        __forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) const override;
        __forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const override;

    private:

        __forceinline __device__ float reflectance(float cosine, float refractionRatio) const;
        __forceinline __device__ float3 refract(float3 incident, float3 normal, float refractionRatio) const;

    };

    inline bool Dielectric::scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const 
    {

        ray.org = (hit.point);
        float3 attenuation = make_float3(1.0f);
        float3 unitDirection = normalize(ray.dir);

        float cosTheta = dot(-unitDirection, hit.normal);
        float refractionRatio;
        if (cosTheta < 0.0f) {
            cosTheta = -cosTheta;
            refractionRatio = _refractionIndex;
        }
        else {
            refractionRatio = 1.0f / _refractionIndex;
        }

        cosTheta = fminf(cosTheta, 1.0f);
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        bool cannotRefract = refractionRatio * sinTheta > 1.0f;
        float3 direction;
        if (cannotRefract || reflectance(cosTheta, refractionRatio) > curand_uniform(rand)) {
            direction = reflect(unitDirection, hit.normal);
            ray.albedo = (_albedo);
        }
        else {
            direction = refract(unitDirection, hit.normal, refractionRatio);
            ray.albedo = (attenuation);
        }

        ray.dir = (direction);
        return true;
    }

    inline bool Dielectric::emitted(Trace::Ray& ray, Trace::Record& hit) const 
    {
        return false; 
    }

    inline float Dielectric::reflectance(float cosine, float refractionRatio) const 
    {
        float r0 = (1.0f - refractionRatio) / (1.0f + refractionRatio);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
    }

    inline float3 Dielectric::refract(float3 incident, float3 normal, float refractionRatio) const 
    {
        float cosTheta = fminf(dot(-incident, normal), 1.0f);
        float3 perp = refractionRatio * (incident + cosTheta * normal);
        float3 parallel = -sqrtf(fabsf(1.0f - length(perp) * length(perp))) * normal;
        return perp + parallel;
    }
};