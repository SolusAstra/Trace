#pragma once
#include "Primitive.h"

namespace Trace {

    class Sphere : public Primitive {

    public:
        float3 _center = make_float3(0.0f, 0.0f, 0.0f);
        float _radius = 0.0f;
        float r2 = 0.0f;
        Trace::Material* _material = nullptr;

    public:

        __forceinline __host__ __device__ Sphere() {}
        __forceinline __host__ __device__ Sphere(float3 center, float radius, Trace::Material* material);

        __forceinline __device__ virtual bool hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) const;

        __forceinline __device__ virtual float3 computeNormal(float3& point) const override { return normalize(point - _center); }
        __forceinline __device__ virtual Trace::Material* getMaterial() const override { return _material; }

    };

    __forceinline __device__ bool Sphere::hit(const Trace::Ray& ray, Trace::Record& hit, float t_max) const {

        float3 SR = ray.org - _center;

        // Quadratic coefficients
        float a = dot(ray.dir, ray.dir);
        float b = dot(SR, ray.dir);
        float c = dot(SR, SR) - r2;

        // Compute discriminant
        float discriminant = b * b - a * c;

        // No collision
        if (discriminant > 0) {

            float root = (-b - sqrtf(discriminant)) / a;
            if (root < t_max && root > 0.001f) {

                this->updatePayload(ray, hit, root);

                return true;
            }

            root = (-b + sqrtf(discriminant)) / a;
            if (root < t_max && root > 0.001f) {

                this->updatePayload(ray, hit, root);

                return true;
            }
        }
        return false;
    }

    inline Sphere::Sphere(float3 center, float radius, Trace::Material* material) {
        _center = center;
        _radius = radius;
        r2 = radius * radius;
        _material = material;
    }

};