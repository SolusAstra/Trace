#pragma once
#include "Primitive.h"

namespace Trace {

    class PrimitiveArray {

    public:

        __host__ __device__ PrimitiveArray() {}
        __forceinline __host__ __device__ PrimitiveArray(Primitive** l, int n) { list = l; list_size = n; }
        __forceinline __device__ bool hit(const Trace::Ray& ray, Trace::Record& hit) const;



    public:
        Primitive** list;
        int list_size;

    };

    __forceinline __device__ bool PrimitiveArray::hit(const Trace::Ray& ray, Trace::Record& hit) const {

        Trace::Record tempHit;
        bool collisionDetected = false;
        float closestHit = FLT_MAX;

        int objectHit = -1;


        // Loop over all objects in the scene
        for (int i = 0; i < list_size; i++) {


            if (list[i]->hit(ray, tempHit, closestHit)) {
                // (Ray,Object) collision detected

                collisionDetected = true;
                closestHit = tempHit.t;
                objectHit = i;

                //list[objectHit]->updatePayload(ray, tempHit, tempHit.t);

                hit.t = tempHit.t;
                hit.point = tempHit.point;
                hit.normal = tempHit.normal;
                hit.material = tempHit.material;
            }
        }

        //if (collisionDetected) {
        //    list[objectHit]->updatePayload(ray, hit, hit.t);
        //}
        return collisionDetected;
    }
};