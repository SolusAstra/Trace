#pragma once
#include "sutil\vec_math.h"
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>

#include "env/Primitive.h"




void sortAlongX(Trace::Primitive* p, std::vector<int>& indices)
{
    std::sort(indices.begin(), indices.end(), [&p](int a, int b) {
        return p->vertex[a].x < p->vertex[b].x;
        });
}


void sortAlongY(Trace::Primitive* p, std::vector<int>& indices)
{
    std::sort(indices.begin(), indices.end(), [&p](int a, int b) {
        return p->vertex[a].y < p->vertex[b].y;
        });
}

void sortAlongZ(Trace::Primitive* p, std::vector<int>& srtIdx)
{
    std::sort(srtIdx.begin(), srtIdx.end(), [&p](int a, int b) {
        return p->vertex[a].z < p->vertex[b].z;
        });
}


struct Idx {
    std::vector<std::vector<int>> data;
    int N;
    int M;

    Idx() {}

    Idx(int N, int M) : N(N), M(M) {
        data.resize(N);
        for (int i = 0; i < N; i++) {
            data[i].resize(M);
            // Initialize data
            std::iota(data[i].begin(), data[i].end(), 0);
        }
    }

    static Idx rankPrimitives(Trace::Primitive* primitive) {

        Trace::Primitive* reducedPrimitive = primitive->reduceToPrimitive();
        Idx rankedIdx(3, reducedPrimitive->N);
        return rankedIdx;
    }

    std::vector<int>& operator()(int axis) { return data[axis]; }
    int& operator()(int axis, int element) { return data[axis][element]; }

    void rank(Trace::Primitive* p);
    void sort(Trace::Primitive* p);

};



inline void Idx::rank(Trace::Primitive* p) {

    // Sort pitives
    sort(p);

    // Temporarily copy 
    Idx temp(*this);
    for (int i = 0; i < M; ++i) {
        temp(0, this->operator()(0, i)) = i;
        temp(1, this->operator()(1, i)) = i;
    }

    // Save ranked indices
    this->data.swap(temp.data);
}



void Idx::sort(Trace::Primitive* p) {
    sortAlongX(p, this->data[0]);
    sortAlongY(p, this->data[1]);
    sortAlongZ(p, this->data[2]);
}
