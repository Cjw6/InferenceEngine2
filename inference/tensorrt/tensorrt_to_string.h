#pragma once

#include <NvInfer.h>
#include <string>

namespace inference {

using namespace nvinfer1;

// Print the shape of a TensorRT tensor
void printShape(Dims64 &dim);

// Print data in the array
template <typename T>
void printArrayRecursion(const T *pArray, Dims64 dim, int iDim, int iStart);

// Get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType);

// Get the string of a TensorRT shape
std::string shapeToString(Dims64 dim);

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType);

// Get the string of a TensorRT data format
std::string formatToString(TensorFormat format);

// Get the string of a TensorRT layer kind
std::string layerTypeToString(LayerType layerType);

// Get the string of a TensorRT tensor location
std::string locationToString(TensorLocation location);

std::string ioModeToString(TensorIOMode iomode);

} // namespace inference