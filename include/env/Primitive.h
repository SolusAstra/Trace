#pragma once
#include <vector>
#include <sutil\vec_math.h>
#include <numeric>

#include "utils/Ray.h"
#include "accel/AABB.h"
#include <cassert>

enum ACCEL_INPUT_TYPE {
    CUSTOM = 0,
    SPHERE = 1,
    TRIANGLE = 2,
    PARTICLE = 3,
    PARTICLE_CUSTOM = 4
};

namespace Trace {
    class dPrimitive {
    public:
        ACCEL_INPUT_TYPE type = ACCEL_INPUT_TYPE::CUSTOM;
        float3* vertex; // Assume externally managed
        float3* normal; // Assume externally managed
        int3* face;     // Assume externally managed
        size_t N;       // Number of vertices
        size_t F;       // Number of faces
        int* matID = nullptr;

        __host__ __device__ dPrimitive() : vertex(nullptr), face(nullptr), N(0), F(0) {}

        ~dPrimitive() {}

        static __host__ void deleteDeviceData(dPrimitive* d_prim) {
            if (d_prim != nullptr) {
                // First, retrieve the dPrimitive object from GPU to access its members
                dPrimitive temp;
                cudaMemcpy(&temp, d_prim, sizeof(dPrimitive), cudaMemcpyDeviceToHost);

                // Free the vertex and face arrays on the GPU
                cudaFree(temp.vertex);
                cudaFree(temp.face);

                // Finally, free the dPrimitive object itself
                cudaFree(d_prim);
            }
        }

        __forceinline __host__ __device__ float3 computeNormal(int idx) const {
            float3 vertexA = vertex[face[idx].x];
            float3 vertexB = vertex[face[idx].y];
            float3 vertexC = vertex[face[idx].z];

            float3 edge1 = vertexB - vertexA;
            float3 edge2 = vertexC - vertexA;

            float3 h = cross(edge2, edge1);
            return normalize(h);
        } // computeNormal()

        __forceinline __host__ __device__ bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const {
            float3 vertexA = vertex[face[idx].x];
            float3 vertexB = vertex[face[idx].y];
            float3 vertexC = vertex[face[idx].z];

            float3 edge1 = vertexB - vertexA;
            float3 edge2 = vertexC - vertexA;
            float3 h = cross(ray.dir, edge2);
            float a = dot(edge1, h);

            if (a > -0.0001f && a < 0.0001f) {
                return false;
            }

            //if (a > -0.00001f && a < 0.00001f) {
            //    return false;
            //}

            //if (fabs(a) < 1) {  // More tolerant to near-parallel cases
            //    return false;
            //}

            float f = 1.0f / a;
            float3 s = ray.org - vertexA;
            float u = f * dot(s, h);
            if (u < 0.0f || u > 1.0f) {
                return false;
            }

            float3 q = cross(s, edge1);
            float v = f * dot(ray.dir, q);
            if (v < 0.0f || u + v > 1.0f) {
                return false;
            }

            //float t = f * dot(edge2, q);
            //if (t > 1e-5 && t < t_max) {  // Include very close intersections
            //    root = t;
            //    return true;
            //}

            float t = f * dot(edge2, q);
            if (t > 0.001f && t < t_max) {
                root = t;
                return true;
            }

            return false;
        }

    };

    class Primitive {

    public:

        ACCEL_INPUT_TYPE type = ACCEL_INPUT_TYPE::CUSTOM;
        size_t N = 0;
        std::vector<float3> vertex = std::vector<float3>();
        std::vector<int3> face = std::vector<int3>();
        std::vector<int> matID = std::vector<int>();

    public:

        virtual bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const = 0;
        virtual AABB surrounding(int objID) const = 0;
        virtual Primitive* reduceToPrimitive() const = 0;
        virtual dPrimitive* createDeviceVersion() const = 0;

        virtual float3 computeNormal(int idx) const = 0;

        __host__ dPrimitive* allocate() const {
            // Allocate memory for dPrimitive on GPU
            dPrimitive* d_prim;
            auto checkCuda = [](cudaError_t result) {
                if (result != cudaSuccess) {
                    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
                    assert(result == cudaSuccess);
                }
            };
            checkCuda(cudaMalloc((void**) &d_prim, sizeof(dPrimitive)));

            // Allocate and copy vertex data to GPU
            float3* d_vertex;
            int3* d_face;
            checkCuda(cudaMalloc((void**) &d_vertex, vertex.size() * sizeof(float3)));
            checkCuda(cudaMalloc((void**) &d_face, face.size() * sizeof(int3)));

            int* d_mat;
            checkCuda(cudaMalloc((void**) &d_mat, matID.size() * sizeof(int)));


            checkCuda(cudaMemcpy(d_vertex, vertex.data(), vertex.size() * sizeof(float3), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(d_face, face.data(), face.size() * sizeof(int3), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(d_mat, matID.data(), matID.size() * sizeof(int), cudaMemcpyHostToDevice));

            // Create a temporary dPrimitive to set up GPU memory
            dPrimitive temp;
            temp.type = this->type;
            temp.vertex = d_vertex;
            temp.face = d_face;
            temp.N = vertex.size();
            temp.F = face.size();
            temp.matID = d_mat;

            // Copy temporary dPrimitive to GPU
            checkCuda(cudaMemcpy(d_prim, &temp, sizeof(dPrimitive), cudaMemcpyHostToDevice));

            return d_prim;
        }

    };
};