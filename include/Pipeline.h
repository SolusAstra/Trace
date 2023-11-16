#pragma once

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "utils\Camera.h"
#include "env\PrimitiveArray.cuh"



namespace Trace {

    struct Pipeline {

        int imageWidth;
        int imageHeight;
        int nSamples;
        PrimitiveArray** d_environment = nullptr;
        Primitive** d_sceneObjects = nullptr;
        curandState* d_rand;
        Camera camera;


        Pipeline(int imageWidth, int imageHeight, int nSamples)
            : imageWidth(imageWidth), imageHeight(imageHeight), nSamples(nSamples) {}





    };

};