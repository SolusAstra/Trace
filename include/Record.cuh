#pragma once
#include <sutil/vec_math.h>


namespace Trace {

    class MaterialConfig;

    struct Record {

        bool emitter = false;

        float t, u, v;
        float3 point;		// Collision point
        float3 normal;		// Normal vector
        
        // Material
        MaterialConfig* material;
    };

};
