#include "pch.h"
#include "Camera.cuh"

using namespace Trace;


Camera::Camera(const float3& position, const float3& target, const float vFOV, const float AR) {

    this->position = position;

    // Calculate the camera frame
    sensorFrame(position, target);

    // Viewport Calculations
    float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
    float halfWidth = AR * halfHeight;
    u *= halfWidth;
    v *= halfHeight;
}

void Camera::update(const float3& position, const float3& target, const float vFOV, const float AR) {

    this->position = position;

    // Calculate the camera frame
    sensorFrame(position, target);

    // Viewport Calculations
    float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
    float halfWidth = AR * halfHeight;
    u *= halfWidth;
    v *= halfHeight;
}