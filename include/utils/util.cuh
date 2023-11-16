#pragma once
#include <sutil/vec_math.h>
#include <fstream>
#include <sstream>
#include <string>

// Load an OBJ file into vertex and index buffers
void LoadObjFile(const char* filename, float3* vertices, int3* indices) {

    
    std::ifstream file(filename);

    if (!file)
    {
        // Failed to open the file
        return;
    }

    int vertexCount = 0;
    int indexCount = 0;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v")
        {
            // Parse vertex position
            float x, y, z;
            iss >> x >> y >> z;
            vertices[vertexCount++] = make_float3(x, y, z);
        }
        else if (token == "f")
        {
            // Parse triangle indices
            int i1, i2, i3;
            iss >> i1 >> i2 >> i3;
            indices[indexCount++] = make_int3(i1 - 1, i2 - 1, i3 - 1);
        }
    }
    
}

