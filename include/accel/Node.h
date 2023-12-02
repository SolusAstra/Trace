#pragma once
#include <random>   
#include <iostream>
#include <unordered_map>
#include "sutil\vec_math.h"

#include "bvh_util.h"
#include "env/Primitive.h"
#include "AABB.h"

// include for std::stack
#include <stack>
#include <queue>
#include <map>


int getRandomAxis() {
    // Static generator to avoid reseeding
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> randInt(0, 3);
    return (int) randInt(gen);
}

void transformIndices(std::vector<int>& subsetIdx, const int axis, Idx& rankIdx) {

    // Get ranked indices along desired axis
    std::vector<int>& rankedAxis = rankIdx.data[(axis == 0) ? 0 : 1];

    // Sort subsetIdx directly using a custom comparison function
    std::sort(subsetIdx.begin(), subsetIdx.end(), [&rankedAxis](int a, int b)
        {
            return rankedAxis[a] < rankedAxis[b];
        });
}

enum TYPE {
    LEAF = 0,
    BRANCH = 1
};

enum BRANCH_STRATEGY {
    SAH,
    MIDPOINT,
    EQUAL_COUNTS
};


class Node {

public:
    TYPE type = BRANCH;



    int depth = 0;
    //size_t nPrimitives = 1;
    int primitiveIdx = -1;
    int branchIdx[2] = { -1, -1 };
    //AABB<T>* box = nullptr;
    bool isLeaf = false;


public:



    Node() {
        //box = new AABB<T>();
    }

    ~Node() {}

    static void generateHierarchy(std::vector<Node>& nodeArray, int& nodeIdx, std::vector<int>& subset, Idx& rankIdx) {

        const int nP = subset.size();

        Node* currentNode = &nodeArray[nodeIdx];
        //currentNode->nPrimitives = nP;

        if (nP == 1) {
            currentNode->isLeaf = true;
            currentNode->primitiveIdx = subset[0];
            return;
        }

        // Sort subset indices along chosen axis
        int axis = getRandomAxis();
        transformIndices(subset, axis, rankIdx);

        auto midpoint = subset.begin() + nP / 2;
        std::vector<int> leftSrtSub(subset.begin(), midpoint);
        std::vector<int> rightSrtSub(midpoint, subset.end());

        // Branching left
        nodeIdx++;
        currentNode->branchIdx[0] = nodeIdx;
        Node::generateHierarchy(nodeArray, nodeIdx, leftSrtSub, rankIdx);

        // Branching right
        nodeIdx++;
        currentNode->branchIdx[1] = nodeIdx;
        Node::generateHierarchy(nodeArray, nodeIdx, rightSrtSub, rankIdx);
    }

};

/* transformIndices()
*  Description: Rearranges the subset indices according to the sorted indices along the desired axis.
*  rankIdx is a struct containing one vector of sorted indices per axis. The sorted indices are computed
*  relative to a vector of vertex positions. The purpose of this function is to take a subset of indices
*  mapped to vertex positions, obtain the ranked indices along the desired axis, and rearrange the
*  subset indices correspondingly.
*
*  Input: N = size of subset, M = size of vertex positions
*  - subsetIdx <std::vector<int>&>  : (N) Subset of indices to be sorted
*  - axis      <const int>          : (1) Axis along which to sort
*  - rankIdx   <sort::Idx&>         : (M) Struct containing sorted indices along each axis
*
*  Output:
*  - subsetIdx <std::vector<int>&>  : (N) Output vector of sorted indices
*
*  Complexity: O(NlogN)
*/
//// Previous version
//// Get ranked indices along desired axis
//std::vector<int>& rankedAxis = rankIdx.data[(axis == 0) ? 0 : 1];
//// Sort subsetIdx directly using a custom comparison function
//std::sort(subsetIdx.begin(), subsetIdx.end(), [&rankedAxis](int a, int b) { return rankedAxis[a] < rankedAxis[b]; });
//// Rearrange sortedOut using subsetIdx as the mapping
//for (int i = 0; i < maxSub; i++) {
//	sortedOut[i] = subsetIdx[i];
//}




//static void generateHierarchy(std::vector<Node>& nodeArray, int& nodeIdx, std::vector<int>& subset, sort::Idx& rankIdx, std::function<int(std::vector<int>)> calcMajorAxis) {
//
//	Node* currentNode = &nodeArray[nodeIdx];
//	currentNode->nPrimitives = subset.size();
//
//	if (subset.size() == 1) {
//		currentNode->isLeaf = true;
//		currentNode->nPrimitives = 1;
//		currentNode->primIdx = subset[0];
//		return;
//	}
//
//	// Sort subset indices along chosen axis
//	//int axis = getRandomAxis();
//
//	int axis = calcMajorAxis(subset);
//
//
//
//	//std::vector<int> sortedSubsetIdx(subset.size());
//	//transformIndices(sortedSubsetIdx, subset, axis, rankIdx);
//
//	transformIndices(subset, axis, rankIdx);
//
//
//
//	std::vector<int> leftSrtSub = std::vector<int>(subset.begin(), subset.begin() + subset.size() / 2);
//	std::vector<int> rightSrtSub = std::vector<int>(subset.begin() + subset.size() / 2, subset.end());
//
//	// Branching left
//	nodeIdx++;
//	currentNode->branchIdx[0] = nodeIdx;
//	Node::generateHierarchy(nodeArray, nodeIdx, leftSrtSub, rankIdx, calcMajorAxis);
//
//	// Branching right
//	nodeIdx++;
//	currentNode->branchIdx[1] = nodeIdx;
//	Node::generateHierarchy(nodeArray, nodeIdx, rightSrtSub, rankIdx, calcMajorAxis);
//}

//void transformIndices2(std::vector<int>& sortedOut, const std::vector<int>& subsetIdx, const int axis, sort::Idx& rankIdx) {
//
//	const int maxSub = subsetIdx.size();
//
//	// Get ranked indices along desired axis
//	std::vector<int>& rankedAxis = rankIdx.data[(axis == 0) ? 0 : 1];
//
//	// Filter and convert subsetIndices using the map
//	std::transform(subsetIdx.cbegin(), subsetIdx.cend(), sortedOut.begin(), [&rankedAxis](auto i) { return rankedAxis[i]; });
//
//	// Sort keys and rearrange values correspondingly
//	std::vector<int> indices(maxSub);
//	std::iota(indices.begin(), indices.end(), 0);
//	std::sort(indices.begin(), indices.end(), [&sortedOut](int a, int b) { return sortedOut[a] < sortedOut[b]; });
//
//	// Rearrange subsetIdx
//	for (int i = 0; i < maxSub; i++) {
//		sortedOut[i] = subsetIdx[indices[i]];
//	}
//}


//#if PRINT_SIM
//std::cout << "Constructing Node " << nodeIdx << std::endl;
//std::cout << "   # of primitives: " << currentNode->nPrimitives << std::endl;
//std::cout << "   subset = [";
//for (auto i : subset) {
//	if (i == subset.back()) {
//		std::cout << i << "]" << std::endl;
//	}
//	else {
//		std::cout << i << ", ";
//	}
//}
//std::cout << "   LEFT " << currentNode->branchIdx[0] << std::endl;
//std::cout << "   RIGHT " << currentNode->branchIdx[1] << std::endl;
//#endif
