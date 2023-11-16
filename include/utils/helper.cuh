#pragma once
#include <curand_kernel.h>
#include <sutil\vec_math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__forceinline __host__ __device__ void wrapSpherical(float& phi, float& theta) {
    phi = fmodf(phi, 2 * M_PI);
    if (phi < 0) { phi += 2 * M_PI; }

    theta = fmodf(theta, M_PI);
    if (theta < 0) { theta += M_PI; }
}


__forceinline __device__ void unitVectorToSpherical(float3 v, float& phi, float& theta) {
    phi = atan2f(v.z, v.x);
    theta = acosf(v.y);

    wrapSpherical(phi, theta);

    phi = (phi + M_PI) / (2 * M_PI);
    theta = theta / M_PI;
}

__forceinline __device__ float3 randFloat3(curandState* rand) {

    float r1 = curand_uniform(rand);
    float r2 = curand_uniform(rand);
    float z = 1.0f - 2.0f * r1;

    float x = cos(2.0f * M_PI * r2) * sqrtf(1.0f - z * z);
    float y = sin(2.0f * M_PI * r2) * sqrtf(1.0f - z * z);
    return normalize(make_float3(x, y, z));
}