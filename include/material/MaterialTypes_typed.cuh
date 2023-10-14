#pragma once
#include <curand_kernel.h>
#include "utils/Ray.cuh"
#include "utils/Record.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ float3 randFloat3(curandState* rand) {

    float r1 = curand_uniform(rand);
    float r2 = curand_uniform(rand);
    float z = 1.0f - 2.0f * r1;

    float x = cos(2.0f * M_PI * r2) * sqrtf(1.0f - z * z);
    float y = sin(2.0f * M_PI * r2) * sqrtf(1.0f - z * z);
    return normalize(make_float3(x, y, z));
}

namespace Albedo {

    struct Solid;
    struct Checker;
    struct Texture;

    //class BaseAlbedoConfig {
    //public:
    //    float3 albedo = make_float3(1.0f, 1.0f, 1.0f);
    //    float3 _even = make_float3(0.0f);
    //    float3 _odd = make_float3(0.0f);

    //    __device__ float3 getAlbedo(const Record& hit);
    //};

    class BaseAlbedoConfig {
    public:
        float3 albedo = make_float3(1.0f, 1.0f, 1.0f);
        float3 _even = make_float3(0.0f);
        float3 _odd = make_float3(0.0f);

        virtual __device__ float3 getAlbedo(const Record& hit) = 0;
    };

    template <typename Albedo_t>
    struct AlbedoConfig : public BaseAlbedoConfig {
        __device__ float3 getAlbedo(const Record& hit) override;
    };

    template <>
    __device__ float3 AlbedoConfig<Albedo::Solid>::getAlbedo(const Record& hit) {
        return albedo;
    }

    template <>
    __device__ float3 AlbedoConfig<Albedo::Checker>::getAlbedo(const Record& hit) {
        float sines = sin(10.0f * hit.point.x) * sin(10.0f * hit.point.y) * sin(10.0f * hit.point.z);
        if (sines < 0.0f) {
            return _odd;
        }
        else {
            return _even;
        }
    }

    template <>
    __device__ float3 AlbedoConfig<Albedo::Texture>::getAlbedo(const Record& hit) {
        // TODO: Implement texture mapping. Utilize code from Application.cu
        return make_float3(0.0f);
    }

};


// Material Effects
namespace MaterialEffect {

    // Terrestrial Effects
    struct DiffuseReflection;
    struct SpecularReflection;
    struct SubsurfaceScattering;
    struct ThinFilmInterference;
    struct Emission;

    // Atmospheric Effects
    struct RayleighScattering;
    struct MieScattering;
    struct Absorption;
    struct MultipleScattering;
    struct VolumetricShadowing;
    struct PhaseFunction;
    struct CloudEmission;
    struct Translucency;
    struct Polarization;
    struct AtmosphericInteraction;

    class BaseMaterialEffectConfig {
    public:
        float roughness = 0.0f;
        float refractionIndex;

        virtual __device__ bool apply(Ray& ray, Record& hit, curandState* rand) = 0;
    };

    template <typename MaterialEffect_t>
    struct MaterialEffectConfig : public BaseMaterialEffectConfig {
        __device__ bool apply(Ray& ray, Record& hit, curandState* rand) override;
    };

    template <>
    __device__ bool MaterialEffectConfig<MaterialEffect::DiffuseReflection>::apply(Ray& ray, Record& hit, curandState* rand) {
        ray.org = hit.point;
        ray.dir = hit.normal;

        if (rand) {
            float3 scatterDirection = normalize(hit.normal + randFloat3(rand));
            if (dot(scatterDirection, hit.normal) > 0.0f) {
                ray.dir = scatterDirection;
            }
        }

        return true;
    }

    template <>
    __device__ bool MaterialEffectConfig<MaterialEffect::SpecularReflection>::apply(Ray& ray, Record& hit, curandState* rand) {

        // Specular reflection with roughness
        ray.org = hit.point;

        if (roughness > 0.0f && rand) {
            ray.dir = normalize(reflect(ray.dir, hit.normal) + roughness * randFloat3(rand));
        }
        else {
            ray.dir = reflect(ray.dir, hit.normal);
        }
        return true;
    }

};


class BaseMaterialConfig {
public:
    virtual __device__ float3 getAlbedo(const Record& hit) = 0;
    virtual __device__ bool applyEffect(Ray& ray, Record& hit, curandState* rand) = 0;

    Albedo::BaseAlbedoConfig* albedoConfig;
    MaterialEffect::BaseMaterialEffectConfig* materialEffectConfig;

    BaseMaterialConfig() : albedoConfig(nullptr), materialEffectConfig(nullptr) {}

    BaseMaterialConfig(Albedo::BaseAlbedoConfig* aConfig, MaterialEffect::BaseMaterialEffectConfig* mConfig)
        : albedoConfig(aConfig), materialEffectConfig(mConfig) {}
};

template <typename Albedo_t, typename MaterialEffect_t>
class MaterialConfig : public BaseMaterialConfig {

public:
    // Concrete material parameters, based on the template types
    Albedo::AlbedoConfig<Albedo_t> specificAlbedo;
    MaterialEffect::MaterialEffectConfig<MaterialEffect_t> specificEffect;

    // Default constructor
    __device__ __host__ MaterialConfig()
        : BaseMaterialConfig(&specificAlbedo, &specificEffect) {}

    // Constructor with configurations
    __device__ __host__ MaterialConfig(const Albedo::AlbedoConfig<Albedo_t>& albedoConfig,
        const MaterialEffect::MaterialEffectConfig<MaterialEffect_t>& effectConfig)
        : BaseMaterialConfig(&specificAlbedo, &specificEffect), specificAlbedo(albedoConfig), specificEffect(effectConfig) {}

    // Overrides for base virtual methods, now utilizing the concrete configurations
    __device__ float3 getAlbedo(const Record& hit) override {
        return specificAlbedo.getAlbedo(hit);
    }

    __device__ bool applyEffect(Ray& ray, Record& hit, curandState* rand) override {
        return specificEffect.apply(ray, hit, rand);
    }
};


/* For a SoA approach, materials are stored as an environment property.
* Say we have 3 materials, then the environment property will be a struct of 3 arrays.
* MaterialConfig<AlbedoConfig, MaterialEffectConfig> materials[3];
* 
* Doing this allows us to have a single material property for all objects in the scene.
* 
* For a ray that hits an object, we can then do:
* 
* MaterialConfig<AlbedoConfig, MaterialEffectConfig> material = materials[hit.material_id];
* 
* // Compute material effects
* material.material_effect.apply(ray, hit, rand);
* 
* // Compute albedo
* float3 albedo = material.albedo.getAlbedo(hit);
*/



