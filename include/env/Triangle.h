#pragma once
#include "Particle.h"


class dTriangle : public dPrimitive {
public:
    dTriangle(const std::vector<float3>& vtx, const std::vector<int3>& fcs) {
        type = ACCEL_INPUT_TYPE::TRIANGLE;
        N = vtx.size();
        F = fcs.size();

        vertex = new float3[N];
        std::copy(vtx.begin(), vtx.end(), vertex);

        face = new int3[F];
        std::copy(fcs.begin(), fcs.end(), face);
    }

    dTriangle(const std::vector<float3>& vtx, const std::vector<int3>& fcs, const std::vector<int>& mat) {
        type = ACCEL_INPUT_TYPE::TRIANGLE;
        N = vtx.size();
        F = fcs.size();

        vertex = new float3[N];
        std::copy(vtx.begin(), vtx.end(), vertex);

        face = new int3[F];
        std::copy(fcs.begin(), fcs.end(), face);

        matID = new int[mat.size()];
        std::copy(mat.begin(), mat.end(), matID);
    }

    virtual ~dTriangle() {}


};

float3 computeBarycenter(const std::vector<float3>& vtx, const std::vector<int3>& idx, int objID) {
    float3 A = vtx[idx[objID].x];
    float3 B = vtx[idx[objID].y];
    float3 C = vtx[idx[objID].z];
    return (A + B + C) / 3.0f;
}

class Triangle : public Primitive {

public:

    // Constructors
    Triangle() {
        this->type = ACCEL_INPUT_TYPE::TRIANGLE;
        this->vertex = std::vector<float3>();
        this->N = 0;

    }

    Triangle(std::vector<float3>& vtx, std::vector<int3>& idx) {
        this->type = ACCEL_INPUT_TYPE::TRIANGLE;
        this->vertex = vtx;
        this->N = idx.size();
        this->face = idx;
    }

    Triangle(std::vector<float3>& vtx, std::vector<int3>& idx, std::vector<int>& mat) {
        this->type = ACCEL_INPUT_TYPE::TRIANGLE;
        this->vertex = vtx;
        this->N = idx.size();
        this->face = idx;
        this->matID = mat;
    }

    __host__ __device__ void getTriangleVertices(int objID, float3& A, float3& B, float3& C) const {
        int3 triIndices = face[objID];
        A = vertex[triIndices.x];
        B = vertex[triIndices.y];
        C = vertex[triIndices.z];
    }

    virtual Primitive* reduceToPrimitive() const override {

        int nFaces = face.size();
        std::vector<float3> barycenter(nFaces);

        for (int k = 0; k < nFaces; k++) {
            barycenter[k] = computeBarycenter(vertex, face, k);
        }

        Particle particlePrimtive = Particle(barycenter);
        return &particlePrimtive;

    }

    virtual bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const override {

        float3 vertexA = vertex[face[idx].x];
        float3 vertexB = vertex[face[idx].y];
        float3 vertexC = vertex[face[idx].z];

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

    virtual AABB surrounding(int objID) const override {
        AABB box;

        int3 triID = face[objID];

        box.min = vertex[triID.x];
        box.max = vertex[triID.x];

        box.min = fminf(box.min, vertex[triID.y]);
        box.max = fmaxf(box.max, vertex[triID.y]);

        box.min = fminf(box.min, vertex[triID.z]);
        box.max = fmaxf(box.max, vertex[triID.z]);

        return box;
    }

    virtual dPrimitive* createDeviceVersion() const override {
        // Create a device version of dParticle
        // This implementation assumes the existence of a CPU counterpart
        // Replace with actual logic to copy data from the CPU version

        // Allocate memory for dTriangle on GPU
        dTriangle* d_triangle;
        cudaMalloc((void**) &d_triangle, sizeof(dTriangle));

        // Allocate and copy vertex data to GPU
        float3* d_vertices;
        cudaMalloc((void**) &d_vertices, vertex.size() * sizeof(float3));
        cudaMemcpy(d_vertices, vertex.data(), vertex.size() * sizeof(float3), cudaMemcpyHostToDevice);

        // Allocate and copy face data to GPU
        int3* d_faces;
        cudaMalloc((void**) &d_faces, face.size() * sizeof(int3));
        cudaMemcpy(d_faces, face.data(), face.size() * sizeof(int3), cudaMemcpyHostToDevice);

        int* d_mat;
        cudaMalloc((void**) &d_mat, matID.size() * sizeof(int));
        cudaMemcpy(d_mat, matID.data(), matID.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Create a temporary dTriangle to set up GPU memory
        dTriangle temp(vertex, face);
        temp.vertex = d_vertices;
        temp.face = d_faces;
        temp.N = vertex.size();
        temp.F = face.size();
        temp.matID = d_mat;

        // Copy temporary dTriangle to GPU
        cudaMemcpy(d_triangle, &temp, sizeof(dTriangle), cudaMemcpyHostToDevice);

        return d_triangle;
    }

};