#pragma once
#include <curand_kernel.h>
#include "Ray.cuh"
#include "Record.cuh"
#include "utils/helper.cuh"



namespace Trace {

    //// Terrestrial Effects
    //struct DiffuseReflection;
    //struct SpecularReflection;
    //struct SubsurfaceScattering;
    //struct ThinFilmInterference;
    //struct Emission;

    //// Atmospheric Effects
    //struct RayleighScattering;
    //struct MieScattering;
    //struct Absorption;
    //struct MultipleScattering;
    //struct VolumetricShadowing;
    //struct PhaseFunction;
    //struct CloudEmission;
    //struct Translucency;
    //struct Polarization;
    //struct AtmosphericInteraction;

    __device__ void LambertianScattering(float3& rayDir, const float3& normal, curandState* rand) {

        // Default behavior
        rayDir = normal;

        // Diffuse scattering
        float3 scatterDirection = normalize(normal + randFloat3(rand));
        if (dot(scatterDirection, normal) > 0.0f) { 
            rayDir = scatterDirection; 
        }
    }

    __device__ void SpecularScattering(float3& rayDir, const float3& normal, float roughness, curandState* rand) {
        rayDir = normalize(reflect(rayDir, normal) + roughness * randFloat3(rand));
    }


    class Material {

    public:
        __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) = 0;

        __device__ virtual void updateTransform(const Transform& transform) = 0;
        //__device__ virtual void computeAlbedo(float3& albedoResult, Trace::Record& hit) = 0;
    };

    class DiffuseReflection : public Material {

    public:
        float3 albedo = make_float3(0.0f);

        __device__ DiffuseReflection() {}

        __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr);

    };

    class SpecularReflection : public Material {

    public:

        float3 albedo = make_float3(0.0f);
        float roughness = 0.0f;

        __device__ SpecularReflection() {}

        __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) override {
            ray.albedo = albedo;
            ray.org = hit.point;
            SpecularScattering(ray.dir, hit.normal, roughness, rand);
            return true;
        };

    };

    class SubsurfaceScattering : public Material {

    };

    class ThinFilmInterference : public Material {

    };

    class Emission : public Material {

    };

    __device__ bool DiffuseReflection::scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand) {
        ray.albedo = albedo;
        ray.org = hit.point;
        LambertianScattering(ray.dir, hit.normal, rand);
        return true;
    };


};

