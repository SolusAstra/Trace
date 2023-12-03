#pragma once

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "utils\Camera.h"
#include "env\Environment.h"



namespace Trace {

    struct Pipeline {

        int imageWidth;
        int imageHeight;
        int nSamples;

        Environment* d_environment = nullptr; // Environment pointer
        curandState* d_rand;
        Camera camera;


        Pipeline(int imageWidth, int imageHeight, int nSamples)
            : imageWidth(imageWidth), imageHeight(imageHeight), nSamples(nSamples) {}





    };

};