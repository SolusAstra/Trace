#pragma once
#include "env/Primitive.h"
//#include "accel/BVH.h"
//#include "accel/BruteForce.h"

#include "utils/Record.h"

namespace Trace {

    class Environment {

    public:
        dPrimitive* primitive = nullptr;

        //Primitive* primitive = nullptr;
        Material* materials = nullptr;

        

        __host__ Environment() {}
        //__host__ Environment(Primitive* prim, Material* mats) : primitive(prim), materials(mats) {}
        __host__ __device__ Environment(dPrimitive* prim, Material* mats) : primitive(prim), materials(mats) {}

    public:

        __forceinline __host__ __device__ bool hit(const Trace::Ray& ray, Trace::Record& payload) {

            int objectIndex = -1;
            bool hitDetected = false;
            float closestHit = FLT_MAX;

            for (int k = 0; k < primitive->N; k++) {

                float root = -1.0f;
                bool primHit = primitive->hit(k, ray, closestHit, root);

                if (primHit && objectIndex == 2964) {
                    int stop = 1 + 1;
                }

                if (primHit && root < closestHit) {
                    hitDetected = true;
                    closestHit = root;
                    objectIndex = k;
                }
            }

            if (hitDetected) {

                if (objectIndex == 3591) {
                    int stop = 1 + 1;
                }


                payload.t = closestHit;
                payload.point = ray.org + payload.t * ray.dir;
                payload.normal = primitive->computeNormal(objectIndex);

                // Offset hit point to avoid floating point bias
                payload.point += payload.normal * 0.00001f;
                if (dot(payload.normal, ray.dir) > 0.0f) {
                    payload.normal = -payload.normal;
                }

                int matID = primitive->matID[objectIndex];
                payload.matID = matID;
            }

            return hitDetected;
        }

        //__forceinline __host__ bool hit(const Trace::Ray& ray, Trace::Record& payload) {

//    int objectIndex = -1;
//    bool hitDetected = false;
//    float closestHit = FLT_MAX;

//    for (int k = 0; k < primitive->N; k++) {

//        float root = -1.0f;
//        bool primHit = primitive->hit(k, ray, closestHit, root);

//        if (primHit && root < closestHit) {
//            hitDetected = true;
//            closestHit = root;
//            objectIndex = k;
//        }
//    }

//    if (hitDetected) {
//        payload.t = closestHit;
//        payload.point = ray.org + payload.t * ray.dir;
//        payload.normal = primitive->computeNormal(objectIndex);

//        // Offset hit point to avoid floating point bias
//        payload.point += payload.normal * 0.0001f;
//        if (dot(payload.normal, ray.dir) > 0.0f) {
//            payload.normal = -payload.normal;
//        }

//        int matID = primitive->matID[objectIndex];
//        payload.matID = matID;
//    }

//    return hitDetected;
//}

        //template <>
        //__forceinline __host__ __device__ bool hit(AccelStruct::BruteForce* linearSearch, const Trace::Ray& ray, Trace::Record& payload) {

        //    Trace::Record tempHit;

        //    bool collisionDetected = false;
        //    float closestHit = FLT_MAX;

        //    int objectHit = -1;

        //    for (int k = 0; k < primitive->N; k++) {

        //        float root = -1.0f;
        //        primitive->hit(k, ray, closestHit, root);

        //        if (root > 0 && root < closestHit) {

        //            collisionDetected = true;
        //            closestHit = root;


        //            payload.t = root;
        //        }

        //    }

        //    return collisionDetected;
        //}


        //template <>
        //inline static bool hit(AccelStruct::BVH* bvh, Primitive* primitive, const Trace::Ray& ray, Trace::Record& payload) {

        //    int stack[64]; // Fixed-size stack for traversal
        //    int stackIndex = 0;
        //    stack[stackIndex++] = 0; // Start with the root node

        //    bool hitSomething = false;
        //    float closestHit = FLT_MAX; // Initialize with maximum float value

        //    while (stackIndex > 0) {
        //        int nodeIdx = stack[--stackIndex]; // Pop a node index from the stack


        //        Node* nodePtr = &bvh->node[nodeIdx];
        //        AABB* bboxPtr = &bvh->bbox[nodeIdx];
        //        //Node* nodePtr = &bvh->node[nodeIdx];
        //        //AABB* bboxPtr = &bvh->bbox[nodeIdx];


        //        if (!bboxPtr->hit(ray, 0, closestHit)) {
        //            continue; // Skip if ray does not hit the bounding box or if box is further than current closest
        //        }

        //        if (nodePtr->isLeaf) {
        //            int primIdx = nodePtr->primitiveIdx;

        //            float root = -1.0f;
        //            bvh->primitive->hit(primIdx, ray, closestHit, root);

        //            //float dist = Triangle::hit(ray, closestSoFar, tri.vertex.data());
        //            if (root > 0 && root < closestHit) {
        //                hitSomething = true;
        //                closestHit = root; // Update closest hit distance

        //                payload.wasHit = true;
        //                payload.primitiveID = primIdx; // Record the index of the closest primitive
        //                payload.t = root;
        //                //payload.path.push_back(nodeIdx); // Optionally record the path
        //            }
        //        }
        //        else {
        //            // Push child nodes onto the stack
        //            stack[stackIndex++] = nodePtr->branchIdx[0]; // Left child
        //            stack[stackIndex++] = nodePtr->branchIdx[1]; // Right child
        //        }
        //    }

        //    return hitSomething;
        //}



    };

}

