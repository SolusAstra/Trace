#pragma once
#include <iostream>
#include "bvh_util.h"
#include "env/Primitive.h"
#include "Node.h"


namespace AccelStruct {

    class BVH {

    public:
        std::vector<Node> node;
        std::vector<AABB> bbox;

        size_t size = 0;
        int depth = 0;

        Trace::Primitive* primitive;

    public:

        BVH() {}
        ~BVH() {}

        void setPtrs();
        void Build(Idx& rankIdx);
        BVH(Trace::Primitive* prim);

    };



    // Kick off recursive build
    void BVH::Build(Idx& rankIdx) {
        int rootIdx = 0;
        std::vector<int> subset = std::vector<int>(rankIdx.M);
        std::iota(subset.begin(), subset.end(), 0);
        Node::generateHierarchy(this->node, rootIdx, subset, rankIdx);
    }

    BVH::BVH(Trace::Primitive* prim) {
        Idx rankIdx = Idx::rankPrimitives(prim);
        this->size = 2 * rankIdx.M - 1;
        this->node = std::vector<Node>(size);
        Build(rankIdx);
    }

    class dBVH {
    public:
        Node* nodePtr;
        AABB* bboxPtr;
        size_t size;
        int depth;
        Trace::dPrimitive* dprimitive; // Assumed to be a deep copy or reference as appropriate

        dBVH(BVH* bvh) {
            if (bvh == nullptr) {
                size = 0;
                depth = 0;
                nodePtr = nullptr;
                bboxPtr = nullptr;
                dprimitive = nullptr;
                return;
            }

            size = bvh->size;
            depth = bvh->depth;


            Trace::dPrimitive* dprim = bvh->primitive->createDeviceVersion();

            dprimitive = dprim; // Adjust this as per the ownership and copying strategy

            nodePtr = new Node[size];
            bboxPtr = new AABB[size];

            std::copy(bvh->node.begin(), bvh->node.end(), nodePtr);
            std::copy(bvh->bbox.begin(), bvh->bbox.end(), bboxPtr);
        }

        ~dBVH() {
            delete[] nodePtr;
            delete[] bboxPtr;
            // Handle primitive deletion or reference decrement here if needed
        }

        // Other methods as required
    };




    void computeDepth(BVH* bvh, int& nodeIdx) {

        // Recursively dive into tree structure and set depth of each node.
        // Compute final depth of tree structure.
        int bvhDepth = 0;
        Node* nodePtr = &bvh->node[nodeIdx];

        if (nodePtr->isLeaf) {
            nodePtr->depth = 0;
            bvhDepth = 0;
        }
        else {
            int leftBranch = nodePtr->branchIdx[0];
            int rightBranch = nodePtr->branchIdx[1];
            computeDepth(bvh, leftBranch);
            computeDepth(bvh, rightBranch);

            // Set depth of current node to 1 + max(left, right)
            nodePtr->depth = 1 + std::max(bvh->node[leftBranch].depth, bvh->node[rightBranch].depth);

            bvhDepth = nodePtr->depth;
        }

        // Set depth of BVH to be maximum depth of tree structure
        bvh->depth = std::max(bvh->depth, bvhDepth);
    }

    void getPath(BVH* bvh, int& nodeIdx, bool& primFound, const int primID, std::vector<int>& path) {

        // Recursively dive into tree structure and set depth of each node.
        // Compute final depth of tree structure.
        Node* nodePtr = &bvh->node[nodeIdx];

        if (nodePtr->isLeaf) {

            if (nodePtr->primitiveIdx == primID) {

                primFound = true;
                path.push_back(nodeIdx);
            }

        }
        else {
            int leftBranch = nodePtr->branchIdx[0];
            int rightBranch = nodePtr->branchIdx[1];
            getPath(bvh, leftBranch, primFound, primID, path);

            if (primFound) {
                path.push_back(leftBranch);
                return;
            }

            bool primRight = false;
            //primFound = false;

            getPath(bvh, rightBranch, primFound, primID, path);

            if (primFound) {
                path.push_back(rightBranch);
                return;
            }
        }

    }

    void computeBoundingBoxes_BVH(BVH* bvh, Trace::Primitive* prim, int& nodeIdx) {

        Node* nodePtr = &bvh->node[nodeIdx];
        AABB* bboxPtr = &bvh->bbox[nodeIdx];

        if (nodePtr->isLeaf) {
            int primIdx = nodePtr->primitiveIdx;

            AABB* bounding = new AABB(prim->surrounding(primIdx));
            bboxPtr->min = bounding->min;
            bboxPtr->max = bounding->max;

            nodeIdx++;
        }
        else {
            // Construct bounding boxes for children
            for (int i = 0; i < 2; i++) {
                int branchIdx = nodePtr->branchIdx[i];
                computeBoundingBoxes_BVH(bvh, prim, branchIdx);
            }

            // Construct bounding box for parent
            int branchIdx0 = nodePtr->branchIdx[0];
            int branchIdx1 = nodePtr->branchIdx[1];

            // Get bounding boxes of children
            AABB* leftBox = &bvh->bbox[branchIdx0];
            AABB* rightBox = &bvh->bbox[branchIdx1];
            AABB* bounding = new AABB(surrounding(*leftBox, *rightBox));
            bboxPtr->min = bounding->min;
            bboxPtr->max = bounding->max;

            nodeIdx++;
        }
    }

    void initBoundingBoxes(BVH* bvh, Trace::Primitive* prim) {

        // Allocate memory for bounding boxes
        bvh->bbox = std::vector<AABB>(bvh->size);

        int nodeIdx = 0;
        computeBoundingBoxes_BVH(bvh, prim, nodeIdx);
        //computeDepth(bvh, prim, nodeIdx);
    }
};