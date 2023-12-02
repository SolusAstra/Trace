#pragma once
#include "Primitive.h"

namespace Trace {

    class Triangle : public Primitive {

    public:
        float3 vertexA = make_float3(0.0f, 0.0f, 0.0f);
        float3 vertexB = make_float3(0.0f, 0.0f, 0.0f);
        float3 vertexC = make_float3(0.0f, 0.0f, 0.0f);
        Trace::Material* _material = nullptr;
        float3 _normal = make_float3(0.0f, 1.0f, 0.0f);

        // precomputed values
        float3 edge1 = make_float3(0.0f, 0.0f, 0.0f);
        float3 edge2 = make_float3(0.0f, 0.0f, 0.0f);
        float3 h = make_float3(0.0f, 0.0f, 0.0f);
        float a = 0.0f;

    public:

        __forceinline __host__ __device__ Triangle() {}
        __forceinline __host__ __device__ Triangle(float3 A, float3 B, float3 C, Trace::Material* material);
        __forceinline __host__ __device__ Triangle(float3 A, float3 B, float3 C, float3 normal, Trace::Material* material);

        __forceinline __device__ virtual bool hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) const;

        __forceinline __device__ virtual float3 computeNormal(float3& point) const override { return _normal; }
        __forceinline __device__ virtual Trace::Material* getMaterial() const override { return _material; }

        __forceinline __host__ __device__ static float3 getNormal(float3 A, float3 B, float3 C);
        __forceinline __host__ __device__ static void rotateBox(float3* vertices, float angleX, float angleY);
        __forceinline __host__ __device__ static void rotateBoxY(float3* vertices, float angle);
        __forceinline __host__ __device__ static void rectangleXY(Primitive** objects, float xMin, float xMax, float yMin, float yMax, float zLevel, Trace::Material* material);
        __forceinline __host__ __device__ static void rectangleXZ(Primitive** objects, float xMin, float xMax, float zMin, float zMax, float yLevel, Trace::Material* material);
        __forceinline __host__ __device__ static void rectangleYZ(Primitive** objects, float yMin, float yMax, float zMin, float zMax, float xLevel, Trace::Material* material);
        __forceinline __host__ __device__ static void rectangle(Primitive** objects, float3* vertices, int3* indices, Trace::Material* material);
        __forceinline __host__ __device__ static void box(Primitive** objects, float3 origin, float2 dimensions, Trace::Material* material);

    };


    __forceinline __device__ bool Triangle::hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) const {
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

        float t = f * dot(edge2, q);
        if (t > 0.001f && t < t_max) {


            this->updatePayload(ray, hit, t);


            return true;
        }
        return false;
    }


    inline Triangle::Triangle(float3 A, float3 B, float3 C, Trace::Material* material)
        :vertexA(A), vertexB(B), vertexC(C), _material(material)
    {
        // precompute edges and cross product
        edge1 = vertexB - vertexA;
        edge2 = vertexC - vertexA;
        h = cross(edge2, edge1);
        _normal = normalize(h);
        a = dot(edge1, h);
    }

    inline Triangle::Triangle(float3 A, float3 B, float3 C, float3 normal, Trace::Material* material)
        : vertexA(A), vertexB(B), vertexC(C), _normal(normal), _material(material)
    {
        // precompute edges and cross product
        edge1 = vertexB - vertexA;
        edge2 = vertexC - vertexA;
        h = cross(edge2, edge1);
        a = dot(edge1, h);
    }

    inline __host__ float3 Triangle::getNormal(float3 A, float3 B, float3 C) {
        float3 edge1 = B - A;
        float3 edge2 = C - A;
        return normalize(cross(edge1, edge2));
    }

    inline __host__ void Triangle::rotateBox(float3* vertices, float angleX, float angleY) {

        // Find the center point of the box
        const float3 center = (vertices[0] + vertices[6]) * 0.5f;

        // Translate the box so that the center point is at the origin
        for (int i = 0; i < 8; ++i) {
            vertices[i] -= center;
        }

        // Rotate the box vertices about the X axis
        const float sinAngleX = sinf(angleX);
        const float cosAngleX = cosf(angleX);
        for (int i = 0; i < 8; ++i) {
            const float3 v = vertices[i];
            vertices[i].y = cosAngleX * v.y + sinAngleX * v.z;
            vertices[i].z = -sinAngleX * v.y + cosAngleX * v.z;
        }

        // Rotate the box vertices about the Y axis
        const float sinAngleY = sinf(angleY);
        const float cosAngleY = cosf(angleY);
        for (int i = 0; i < 8; ++i) {
            const float3 v = vertices[i];
            vertices[i].x = cosAngleY * v.x + sinAngleY * v.z;
            vertices[i].z = -sinAngleY * v.x + cosAngleY * v.z;
        }

        // Translate the box back to its original position
        for (int i = 0; i < 8; ++i) {
            vertices[i] += center;
        }
    }

    inline __host__ void Triangle::rotateBoxY(float3* vertices, float angle) {

        // Find the center point of the box
        const float3 center = (vertices[0] + vertices[6]) * 0.5f;

        // Translate the box so that the center point is at the origin
        for (int i = 0; i < 8; ++i) {
            vertices[i] -= center;
        }

        // Rotate the box vertices about the y-axis
        const float sinAngle = sinf(angle);
        const float cosAngle = cosf(angle);
        for (int i = 0; i < 8; ++i) {
            const float3 v = vertices[i];
            vertices[i].x = cosAngle * v.x + sinAngle * v.z;
            vertices[i].z = -sinAngle * v.x + cosAngle * v.z;
        }

        // Translate the box back to its original position
        for (int i = 0; i < 8; ++i) {
            vertices[i] += center;
        }
    }

    inline __host__ void Triangle::rectangleXY(Primitive** objects, float xMin, float xMax, float yMin, float yMax, float zLevel, Trace::Material* material) {

        // Define the vertices
        float3* vertices = new float3[4];
        vertices[0] = make_float3(xMin, yMin, zLevel);
        vertices[1] = make_float3(xMin, yMax, zLevel);
        vertices[2] = make_float3(xMax, yMax, zLevel);
        vertices[3] = make_float3(xMax, yMin, zLevel);


        // Define indices
        int3* indices = new int3[2];
        indices[0] = make_int3(0, 1, 2);
        indices[1] = make_int3(0, 2, 3);

        // Construct the triangles
        rectangle(objects, vertices, indices, material);
    }

    inline __host__ void Triangle::rectangleXZ(Primitive** objects, float xMin, float xMax, float zMin, float zMax, float yLevel, Trace::Material* material) {

        // Define the vertices
        float3* vertices = new float3[4];
        vertices[0] = make_float3(xMin, yLevel, zMin);
        vertices[1] = make_float3(xMin, yLevel, zMax);
        vertices[2] = make_float3(xMax, yLevel, zMax);
        vertices[3] = make_float3(xMax, yLevel, zMin);

        // Define indices
        int3* indices = new int3[2];
        indices[0] = make_int3(0, 1, 2);
        indices[1] = make_int3(0, 2, 3);

        // Construct the triangles
        rectangle(objects, vertices, indices, material);
    }

    // YZ rectangle

    inline __host__ void Triangle::rectangleYZ(Primitive** objects, float yMin, float yMax, float zMin, float zMax, float xLevel, Trace::Material* material) {

        // Define the vertices
        float3* vertices = new float3[4];
        vertices[0] = make_float3(xLevel, yMin, zMin);
        vertices[1] = make_float3(xLevel, yMin, zMax);
        vertices[2] = make_float3(xLevel, yMax, zMax);
        vertices[3] = make_float3(xLevel, yMax, zMin);

        // Define indices
        int3* indices = new int3[2];
        indices[0] = make_int3(0, 1, 2);
        indices[1] = make_int3(0, 2, 3);

        // Construct the triangles
        rectangle(objects, vertices, indices, material);
    }

    inline __host__ void Triangle::rectangle(Primitive** objects, float3* vertices, int3* indices, Trace::Material* material) {

        // Get vertices
        float3 vertexA = vertices[indices[0].x];
        float3 vertexB = vertices[indices[0].y];
        float3 vertexC = vertices[indices[0].z];
        float3 vertexD = vertices[indices[1].z];

        // Initialize the triangles
        Triangle* triangleA = new Triangle(vertexA, vertexB, vertexC, material);
        Triangle* triangleB = new Triangle(vertexA, vertexC, vertexD, material);

        // Add triangles to objects
        objects[0] = triangleA;
        objects[1] = triangleB;
    }

    inline __host__ void Triangle::box(Primitive** objects, float3 origin, float2 dimensions, Trace::Material* material) {

        float widthMinorAxis = dimensions.x;
        float heightMajorAxis = dimensions.y;

        float3 vertices[8];
        vertices[0] = make_float3(origin.x - widthMinorAxis / 2.0f, origin.y, origin.z - widthMinorAxis / 2.0f);  // Front bottom left
        vertices[1] = make_float3(origin.x - widthMinorAxis / 2.0f, origin.y, origin.z + widthMinorAxis / 2.0f);  // Front top left
        vertices[2] = make_float3(origin.x + widthMinorAxis / 2.0f, origin.y, origin.z + widthMinorAxis / 2.0f);  // Front top right
        vertices[3] = make_float3(origin.x + widthMinorAxis / 2.0f, origin.y, origin.z - widthMinorAxis / 2.0f);  // Front bottom right
        vertices[4] = make_float3(origin.x - widthMinorAxis / 2.0f, origin.y + heightMajorAxis, origin.z - widthMinorAxis / 2.0f);  // Back bottom left
        vertices[5] = make_float3(origin.x - widthMinorAxis / 2.0f, origin.y + heightMajorAxis, origin.z + widthMinorAxis / 2.0f);  // Back top left
        vertices[6] = make_float3(origin.x + widthMinorAxis / 2.0f, origin.y + heightMajorAxis, origin.z + widthMinorAxis / 2.0f);  // Back top right
        vertices[7] = make_float3(origin.x + widthMinorAxis / 2.0f, origin.y + heightMajorAxis, origin.z - widthMinorAxis / 2.0f);  // Back bottom right

        rotateBoxY(vertices, 3.14f / 4.0f);

        // Define the indices for each of the cube's faces
        int3 indices[12];
        indices[0] = make_int3(0, 1, 2);
        indices[1] = make_int3(0, 2, 3);
        indices[2] = make_int3(7, 6, 5);
        indices[3] = make_int3(7, 5, 4);
        indices[4] = make_int3(4, 5, 1);
        indices[5] = make_int3(4, 1, 0);
        indices[6] = make_int3(3, 2, 6);
        indices[7] = make_int3(3, 6, 7);
        indices[8] = make_int3(4, 0, 3);
        indices[9] = make_int3(4, 3, 7);
        indices[10] = make_int3(1, 5, 6);
        indices[11] = make_int3(1, 6, 2);

        float3 frontVertices[] = { vertices[indices[0].x], vertices[indices[0].y], vertices[indices[0].z], vertices[indices[1].z] };
        int3 frontIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[0], frontVertices, frontIndices, material);

        // Create triangles for the back face
        float3 backVertices[] = { vertices[indices[2].x], vertices[indices[2].y], vertices[indices[2].z], vertices[indices[3].z] };
        int3 backIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[2], backVertices, backIndices, material);

        // Create triangles for the left face
        float3 leftVertices[] = { vertices[indices[4].x], vertices[indices[4].y], vertices[indices[4].z], vertices[indices[5].z] };
        int3 leftIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[4], leftVertices, leftIndices, material);

        // Create triangles for the right face
        float3 rightVertices[] = { vertices[indices[6].x], vertices[indices[6].y], vertices[indices[6].z], vertices[indices[7].z] };
        int3 rightIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[6], rightVertices, rightIndices, material);

        // Create triangles for the bottom face
        float3 bottomVertices[] = { vertices[indices[8].x], vertices[indices[8].y], vertices[indices[8].z], vertices[indices[9].z] };
        int3 bottomIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[8], bottomVertices, bottomIndices, material);

        // Create triangles for the top face
        float3 topVertices[] = { vertices[indices[10].x], vertices[indices[10].y], vertices[indices[10].z], vertices[indices[11].z] };
        int3 topIndices[] = { make_int3(0, 1, 2), make_int3(0, 2, 3) };
        Triangle::rectangle(&objects[10], topVertices, topIndices, material);

    }
};