#pragma once
#include "Sphere.cuh"




namespace Trace {

    class Environment {

    public:
        Trace::Sphere* sphere;
        int N;

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
            for (int i = 0; i < N; i++) {

                float root = -1.0f;
                bool hitFound = sphere[i].hitRoot(ray, root, closestHit);
                if (hitFound) {
                    wasHit = true;
                    closestHit = root;
                    objectIndex = i;
                }

            }
            if (wasHit) {
                ray.t = closestHit;
                hit.point = ray.getCast();
                hit.normal = normalize(hit.point - sphere[objectIndex].center);

                // Offset hit point to avoid floating point bias
                hit.point += hit.normal * 0.0001f;
                if (dot(ray.dir, hit.normal) > 0) {
                    hit.normal = -hit.normal;
                }

                ray.albedo = make_float3(1.0f, 0.0f, 0.0f);

                // Get the material from the pool
                Material* mat = sphere[objectIndex]._material;
                mat->scatter(ray, hit, rand);

                if (hit.emitter) {
                    ray.albedo = make_float3(1.0f, 1.0f, 1.0f);
                }


                return true;
            }

            return false;
        }



    };




};