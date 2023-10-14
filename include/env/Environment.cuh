#pragma once
#include "env/SphereSOA.cuh"
#include "material/MaterialTypes.cuh"



namespace Trace {

    class Environment {

    public:
        Trace::SphereSoA* sphereSoA;
        Trace::Material** globalMaterialPool;

    public:

        __host__ __device__ Environment() {}

        // Allocate environment and return device pointer
        __host__ static Environment* allocate() {
            Environment* d_env;
            cudaMalloc(&d_env, sizeof(Environment));
            return d_env;
        }

        __device__ bool hit(Trace::Ray& ray, Trace::Record& hit, curandState* rand) const {

            bool wasHit = false;
            float closestHit = FLT_MAX;
            int objectIndex = -1;


            // Check if ray hits any of the spheres
            for (int i = 0; i < sphereSoA->size; i++) {

                float root = -1.0f;
                bool hitFound = sphereSoA->hitRoot(i, ray, root, closestHit);
                if (hitFound) {
                    wasHit = true;
                    closestHit = root;
                    objectIndex = i;
                }

            }
            if (wasHit) {
                ray.t = closestHit;
                hit.point = ray.getCast();
                hit.normal = normalize(hit.point - sphereSoA->center[objectIndex]);

                // Offset hit point to avoid floating point bias
                hit.point += hit.normal * 0.0001f;
                if (dot(ray.dir, hit.normal) > 0) {
                    hit.normal = -hit.normal;
                }

                Trace::SphereSoA::get_sphere_uv(hit.point, hit.u, hit.v);

                // Get the material from the pool
                Trace::Material* material = globalMaterialPool[objectIndex];
                material->scatter(ray, hit, rand);

                if (hit.emitter) {
                    ray.albedo = make_float3(1.0f, 1.0f, 1.0f);
                }


                return true;
            }

            return false;
        }



    };




};