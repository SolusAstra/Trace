#pragma once
#include "Primitive.h"
#include "Ray.h"


namespace Trace {

    class Triangle : public Primitive {

    public:
        float3* v = nullptr;
        int3* f = nullptr;
        float3* n = nullptr;
        int* matID = nullptr;
        int nFaces = 0;

    public:


        __forceinline __host__ __device__ Triangle() : Primitive(TRIANGLE) {}
        __forceinline __host__ __device__ Triangle(float3* vertex, int nVertices, Material* material) : Primitive(TRIANGLE, vertex, nVertices, material) {}


        __forceinline __host__ __device__ Triangle(float3* vertex, int3* face, int nFaces) : v(vertex), f(face), nFaces(nFaces) {
            this->type = TRIANGLE;
            this->N = nFaces;

        }

        __forceinline __host__ __device__ Triangle(float3* vertex, int3* face, int nFaces, int* matID) : v(vertex), f(face), nFaces(nFaces) {
            this->type = TRIANGLE;
            this->N = nFaces;
            this->index = matID;
        }


        __forceinline __host__ __device__ virtual bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const override {

            float3 vertexA = v[f[idx].x];
            float3 vertexB = v[f[idx].y];
            float3 vertexC = v[f[idx].z];

            float3 edge1 = vertexB - vertexA;
            float3 edge2 = vertexC - vertexA;
            float3 h = cross(ray.dir, edge2);
            float a = dot(edge1, h);

            if (a > -0.00001f && a < 0.00001f) {
                return false;
            }

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

            root = f * dot(edge2, q);
            if (root > 0.001f && root < t_max) {
                return true;
            }

            return false;

        }


        __forceinline __device__ virtual void updatePayload(int idx, const Trace::Ray& ray, Trace::Record& hit, float root) const override {

            hit.t = root;
            hit.point = ray.org + hit.t * ray.dir;

            hit.normal = computeNormal(idx);

            // Offset hit point to avoid floating point bias
            hit.point += hit.normal * 0.0001f;
            if (dot(hit.normal, ray.dir) > 0.0f) {
                hit.normal = -hit.normal;
            }

            //hit.matID = matID[idx];
           //hit.material = _material;

        }

        __forceinline __device__ virtual float3 computeNormal(int idx) const override {

            float3 vertexA = v[f[idx].x];
            float3 vertexB = v[f[idx].y];
            float3 vertexC = v[f[idx].z];

            float3 edge1 = vertexB - vertexA;
            float3 edge2 = vertexC - vertexA;

            return normalize(cross(edge1, edge2));
        } // computeNormal()



    };

};


