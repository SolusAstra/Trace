#pragma once
#include <curand_kernel.h>

#include "utils/Ray.h"
#include "utils/Record.h"
#include "utils/helper.cuh"

enum MATERIAL_TYPE {
    LAMBERTIAN = 0,
    REFLECTIVE = 1,
    EMITTER = 2

};

namespace Trace {

    class Material {

    public:
        MATERIAL_TYPE type;
        float4 prop = make_float4(0.0f);

    public:

        Material() {}
        Material(MATERIAL_TYPE type, float3 color) : type(type), prop(make_float4(color, 0.0f)) {}
        Material(MATERIAL_TYPE type, float3 color, float roughness) : type(type), prop(make_float4(color, roughness)) {}


        //__host__ static Material** allocate(int N) {
        //    Material** d_material;
        //    cudaMalloc((void**) &d_material, N * sizeof(Material*));
        //    return d_material;
        //}

        //__forceinline bool scatter(Trace::Ray& ray, Trace::Record& hit) const {

        //    switch (type) {
        //        case (MATERIAL_TYPE::LAMBERTIAN):
        //        {
        //            // Set attributes of outgoing ray
        //            ray.org = hit.point;
        //            ray.albedo = make_float3(prop);
        //            ray.dir = (hit.normal);

        //            // Lambertian scattering
        //            if (rand) {
        //                float3 scatterDirection = normalize(hit.normal + randFloat3());
        //                if (dot(scatterDirection, hit.normal) > 0.0f) { ray.dir = (scatterDirection); }
        //            }

        //            return true;
        //            break;
        //        }
        //        case (MATERIAL_TYPE::REFLECTIVE):
        //        {
        //            // Set attributes of outgoing ray
        //            ray.org = hit.point;
        //            ray.albedo = make_float3(prop);

        //            if (prop.w > 0.0f && rand) {
        //                ray.dir = (normalize(reflect(ray.dir, hit.normal) + prop.w * randFloat3()));
        //            }
        //            else {
        //                ray.dir = (reflect(ray.dir, hit.normal));
        //            }

        //            return true;
        //            break;
        //        }
        //        case (MATERIAL_TYPE::EMITTER):
        //        {
        //            return false;
        //            break;
        //        }
        //    }

        //}

        //__forceinline bool emitted(Trace::Ray& ray, Trace::Record& hit) const {

        //    switch (type) {
        //        case (MATERIAL_TYPE::LAMBERTIAN):
        //        {
        //            return false;
        //            break;
        //        }
        //        case (MATERIAL_TYPE::REFLECTIVE):
        //        {
        //            return false;
        //            break;
        //        }
        //        case (MATERIAL_TYPE::EMITTER):
        //        {
        //            ray.albedo = make_float3(prop);
        //            return true;
        //            break;
        //        }
        //    }

        //}



        __forceinline __device__ bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) const {

            switch (type) {
                case (MATERIAL_TYPE::LAMBERTIAN):
                {
                    // Set attributes of outgoing ray
                    ray.org = hit.point;
                    ray.albedo = make_float3(prop);
                    ray.dir = (hit.normal);

                    // Lambertian scattering
                    if (rand) {
                        float3 scatterDirection = normalize(hit.normal + randFloat3(rand));
                        if (dot(scatterDirection, hit.normal) > 0.0f) { ray.dir = (scatterDirection); }
                    }

                    return true;
                    break;
                }
                case (MATERIAL_TYPE::REFLECTIVE):
                {
                    // Set attributes of outgoing ray
                    ray.org = hit.point;
                    ray.albedo = make_float3(prop);

                    if (prop.w > 0.0f && rand) {
                        ray.dir = normalize((normalize(reflect(ray.dir, hit.normal) + prop.w * randFloat3(rand))));
                    }
                    else {
                        ray.dir = normalize((reflect(ray.dir, hit.normal)));
                    }

                    return true;
                    break;
                }
                case (MATERIAL_TYPE::EMITTER):
                {
                    return false;
                    break;
                }
            }

        }

        __forceinline __device__ bool emitted(Trace::Ray& ray, Trace::Record& hit) const {

            switch (type) {
                case (MATERIAL_TYPE::LAMBERTIAN):
                {
                    return false;
                    break;
                }
                case (MATERIAL_TYPE::REFLECTIVE):
                {
                    return false;
                    break;
                }
                case (MATERIAL_TYPE::EMITTER):
                {
                    ray.albedo = make_float3(prop);
                    return true;
                    break;
                }
            }

        }





        //__forceinline __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) const = 0;
        //__forceinline __device__ virtual bool emitted(Trace::Ray& ray, Trace::Record& hit) const = 0;

    };

};