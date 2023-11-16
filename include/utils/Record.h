#pragma once
#include <sutil/vec_math.h>

namespace Trace {

    struct Material;

    struct Record {

        Material* material;
        float3 point;		// Collision point
        float3 normal;		// Normal vector
        float t;			// Ray parameter

    };

};