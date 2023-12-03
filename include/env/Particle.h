#pragma once
#include "Primitive.h"

/* It might be easiest to define multiple particle types Particle2, Particle3
 * and then use a template to define Particle<T> where T is either Particle2 or Particle3
 * * This way, we can use the same BVH class for both 2D and 3D particles
*/

namespace Trace {

    class dParticle : public dPrimitive {

    public:
        dParticle(const std::vector<float3>& vtx) {
            type = ACCEL_INPUT_TYPE::PARTICLE;
            N = vtx.size();
            vertex = new float3[N];
            std::copy(vtx.begin(), vtx.end(), vertex);
        }
    };

    class Particle : public Primitive {

    public:
        Particle() {
            this->vertex = std::vector<float3>();
            this->N = 0;
        }

        Particle(const std::vector<float3>& vtx) {
            this->vertex = vtx;
            this->N = vtx.size();
        }

        virtual Primitive* reduceToPrimitive() const override {
            Particle particlePrimtive = Particle(vertex);
            return &particlePrimtive;
        }

        virtual bool hit(int idx, const Trace::Ray& ray, float t_max, float& root) const override {
            root -= 1.0f;
            return true;
        }

        virtual AABB surrounding(int objID) const override {
            return AABB(vertex[objID], vertex[objID]);
        }

        virtual float3 computeNormal(int idx) const override {
            return make_float3(0.0f, 1.0f, 0.0f);
        }

        virtual dPrimitive* createDeviceVersion() const override {
            // Create a device version of dParticle
            // This implementation assumes the existence of a CPU counterpart
            // Replace with actual logic to copy data from the CPU version

            return new dParticle(vertex);
        }

    };

};