#pragma once
#include "utils/Ray.h"
#include "utils/Record.h"
#include <vector>
#include <numeric>

//namespace Sphere {
//    __forceinline __device__ float hit(const Trace::Ray& ray, Trace::Record& hit, float t_max, float3* geo_data);
//};
//
//namespace Triangles {
//    __forceinline __device__ float hit(const Trace::Ray& ray, Trace::Record& hit, float t_max, float3* geo_data);
//};


namespace Trace {

    enum INPUT_TYPE {
        CUSTOM = 0,
        SPHERE = 1,
        TRIANGLE = 2,
        PARTICLE = 3,
        PARTICLE_CUSTOM = 4
    };


    class Primitive {

    public:

        // Primitive identifiers
        INPUT_TYPE type = INPUT_TYPE::CUSTOM;

        // Primitive data
        int N;
        float3* vertex = nullptr;
        int* index = nullptr;
        Material* _material = nullptr;

        // Constructors
        __forceinline __host__ __device__ Primitive() : type(INPUT_TYPE::CUSTOM), N(0) {}
        __forceinline __host__ __device__ Primitive(INPUT_TYPE type) : type(type), N(0) {}

        __forceinline __host__ __device__ Primitive(INPUT_TYPE type, float3* vertices, int numVertices, Material* material)
            : type(type), N(numVertices), vertex(vertices), index(nullptr), _material(material) {}


        __forceinline __host__ __device__ virtual bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const = 0;

        __forceinline __device__ virtual void updatePayload(int idx, const Trace::Ray& ray, Trace::Record& hit, float root) const = 0;
        __forceinline __device__ virtual float3 computeNormal(int idx) const = 0;

        __host__ static Primitive** allocate(int N) {

            Trace::Primitive** d_sceneObjects;
            cudaMalloc((void**) &d_sceneObjects, N * sizeof(Trace::Primitive*));
            return d_sceneObjects;

        }

        //__forceinline __host__ __device__ Primitive() : type(INPUT_TYPE::CUSTOM)  {
        //    for (int i = 0; i < 3; ++i) {
        //        vertex[i] = make_float3(0.0f);
        //    }
        //}
        //__forceinline __device__ Primitive(INPUT_TYPE type, Material* material) : type(type), _material(material) {
        //    for (int i = 0; i < 3; ++i) {
        //        vertex[i] = make_float3(0.0f);
        //    }
        //}
        //__forceinline __device__ Primitive(INPUT_TYPE type, float3* vtx, Material* material) : type(type), _material(material) {
        //    for (int i = 0; i < 3; ++i) {
        //        vertex[i] = vtx[i];
        //    }
        //}

        //__forceinline __device__ static Primitive* Sphere(float3 center, float radius, Trace::Material* material) {

        //    Primitive* prim = new Primitive(INPUT_TYPE::SPHERE, material);

        //    if (!prim) return nullptr; // Check for successful allocation

        //    prim->vertex[0] = center;
        //    prim->vertex[1] = make_float3(radius, 0, 0);

        //    return prim;
        //}

        //__forceinline __device__ static Primitive* Triangle(float3 A, float3 B, float3 C, Trace::Material* material) {


        //    Primitive* prim = new Primitive(INPUT_TYPE::TRIANGLE, material);

        //    if (!prim) return nullptr; // Check for successful allocation

        //    prim->vertex[0] = A;
        //    prim->vertex[1] = B;
        //    prim->vertex[2] = C;

        //    return prim;
        //}

        //__forceinline __device__ virtual void updatePayload(const Trace::Ray& ray, Trace::Record& hit, float root) const {

        //    hit.t = root;
        //    hit.point = ray.org + hit.t * ray.dir;

        //    hit.normal = computeNormal(hit.point);

        //    // Offset hit point to avoid floating point bias
        //    hit.point += hit.normal * 0.0001f;
        //    if (dot(hit.normal, ray.dir) > 0.0f) {
        //        hit.normal = -hit.normal;
        //    }

        //    hit.material = _material;

        //}


        //__forceinline __device__ bool hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) {


        //    float root = Triangles::hit(ray, hit, t_max, vertex);

        //    if (root > 0.0f) {
        //        updatePayload(ray, hit, root);
        //        return true;
        //    }

        //    return false;
        //}

        //__forceinline __device__ float3 computeNormal(float3& point) const {
        //    float3 edge1 = vertex[1] - vertex[0];
        //    float3 edge2 = vertex[2] - vertex[0];

        //    return normalize(cross(edge1, edge2));
        //} // computeNormal()

    }; // class Primitive

}; // namespace Trace

//__forceinline __device__ float Triangles::hit(const Trace::Ray& ray, Trace::Record& hit, float t_max, float3* geo_data) {
//
//    float3 vertexA = geo_data[0];
//    float3 vertexB = geo_data[1];
//    float3 vertexC = geo_data[2];
//
//    float3 edge1 = vertexB - vertexA;
//    float3 edge2 = vertexC - vertexA;
//    float3 h = cross(ray.dir, edge2);
//    float a = dot(edge1, h);
//
//    if (a > -0.00001f && a < 0.00001f) {
//        return -1.0f;
//    }
//
//    float f = 1.0f / a;
//    float3 s = ray.org - vertexA;
//    float u = f * dot(s, h);
//    if (u < 0.0f || u > 1.0f) {
//        return -1.0f;
//    }
//
//    float3 q = cross(s, edge1);
//    float v = f * dot(ray.dir, q);
//    if (v < 0.0f || u + v > 1.0f) {
//        return -1.0f;
//    }
//
//    float root = f * dot(edge2, q);
//    if (root > 0.001f && root < t_max) {
//        return root;
//    }
//
//    return -1.0f;
//}




//__forceinline __device__ float Sphere::hit(
//    const Trace::Ray& ray, Trace::Record& hit, float t_max,
//    float3* geo_data) {
//
//    float3 center = geo_data[0];
//    float radius = geo_data[1].x;
//
//    float3 SR = ray.org - center;
//    float r2 = radius * radius;
//
//    // Quadratic coefficients
//    float a = dot(ray.dir, ray.dir);
//    float b = dot(SR, ray.dir);
//    float c = dot(SR, SR) - r2;
//
//    // Compute discriminant
//    float discriminant = b * b - a * c;
//
//    // No collision
//    if (discriminant > 0) {
//
//
//
//        float root = (-b - sqrtf(discriminant)) / a;
//        if (root < t_max && root > 0.001f) {
//            return root;
//        }
//
//        root = (-b + sqrtf(discriminant)) / a;
//        if (root < t_max && root > 0.001f) {
//            return root;
//        }
//    }
//
//    return -1.0f;
//}

//__forceinline __host__ Primitive(INPUT_TYPE type, std::vector<float3>& vtx)
//    : Primitive(type, vtx.size(), vtx.data(), new int[vtx.size()]) 
//{
//    std::iota(this->index, this->index + vtx.size(), 0);
//}

//__forceinline __host__ Primitive(INPUT_TYPE type, std::vector<float3>& vtx, Material* material)
//    : Primitive(type, vtx.size(), vtx.data(), new int[vtx.size()])
//{
//    std::iota(this->index, this->index + vtx.size(), 0);
//    this->_material = material;
//}

//__forceinline __host__ explicit Primitive(INPUT_TYPE type, size_t N, float3* vtx, int* idx)
//    : type(type), N(N), vertex(vtx), index(idx) {}

//__forceinline __host__ explicit Primitive(INPUT_TYPE type, std::vector<float3>& vtx, std::vector<int>& idx)
//    : Primitive(type, vtx.size(), vtx.data(), idx.data()) {}