#pragma once
#include <sutil\vec_math.h>
#include <vector>
#include "utils\Ray.h"

class AABBBase {
public:
    virtual ~AABBBase() {}
    // Define common interface for AABB operations
};


class AABB : public AABBBase {
public:
    float3 min = make_float3(0.0f);
    float3 max = make_float3(0.0f);

    AABB() {}
    AABB(const float3& min, const float3& max) : min(min), max(max) {}

    __forceinline __host__ __device__ bool hit(const Trace::Ray r, float t_min, float t_max) {

        // For each axis x, y, z
        float t0, t1;

        // X-axis
        float invDx = 1.0f / r.dir.x;
        t0 = (min.x - r.org.x) * invDx;
        t1 = (max.x - r.org.x) * invDx;
        if (invDx < 0.0f) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;

        // Y-axis
        float invDy = 1.0f / r.dir.y;
        t0 = (min.y - r.org.y) * invDy;
        t1 = (max.y - r.org.y) * invDy;
        if (invDy < 0.0f) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;

        // Z-axis
        float invDz = 1.0f / r.dir.z;
        t0 = (min.z - r.org.z) * invDz;
        t1 = (max.z - r.org.z) * invDz;
        if (invDz < 0.0f) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;

        return true;


    }
};


inline int longestAxis(const AABB& box) {
    // Compute longest axis of a bounding box (specific to float 3)
    float3 diff = box.max - box.min;
    if (diff.x >= diff.y && diff.x >= diff.z) {
        return 0; // x-axis is longest
    }
    else if (diff.y >= diff.x && diff.y >= diff.z) {
        return 1; // y-axis is longest
    }
    else {
        return 2; // z-axis is longest
    }

}

// Create a bounding box that surrounds a single point

inline AABB surrounding(const float3& point) {
    AABB box(point, point);
    return box;
}

inline AABB surrounding(const AABB& boxA, const AABB& boxB) {
    return AABB(fminf(boxA.min, boxB.min), fmaxf(boxA.max, boxB.max));
}

// Create a bounding box that surrounds two points
inline AABB surrounding(const std::vector<float3>& points) {
    AABB box;
    box.min = points[0];
    box.max = points[0];
    for (int i = 1; i < points.size(); i++) {
        box.min = fminf(box.min, points[i]);
        box.max = fmaxf(box.max, points[i]);
    }
    return box;
}
