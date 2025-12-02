#include "tensorrt_to_string.h"

#include <iostream>

namespace inference {

using namespace nvinfer1;

// Print the shape of a TensorRT tensor
void printShape(Dims64 &dim) {
  std::cout << "[";
  for (int i = 0; i < dim.nbDims; ++i) {
    std::cout << dim.d[i] << ", ";
  }
  std::cout << "]" << std::endl;
  return;
}

// Get the string of a TensorRT shape
std::string shapeToString(Dims64 dim) {
  std::string output("(");
  if (dim.nbDims == 0) {
    return output + std::string(")");
  }
  for (int i = 0; i < dim.nbDims - 1; ++i) {
    output += std::to_string(dim.d[i]) + std::string(", ");
  }
  output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
  return output;
}

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType) {
  switch (dataType) {
  case DataType::kFLOAT:
    return std::string("FP32 ");
  case DataType::kHALF:
    return std::string("FP16 ");
  case DataType::kINT8:
    return std::string("INT8 ");
  case DataType::kINT32:
    return std::string("INT32");
  case DataType::kBOOL:
    return std::string("BOOL ");
  case DataType::kUINT8:
    return std::string("UINT8");
  case DataType::kFP8:
    return std::string("FP8  ");
  case DataType::kINT64:
    return std::string("INT64");
  default:
    return std::string("Unknown");
  }
}

// Get the string of a TensorRT data format
std::string formatToString(TensorFormat format) {
  switch (format) {
  case TensorFormat::kLINEAR:
    return std::string("LINE ");
  case TensorFormat::kCHW2:
    return std::string("CHW2 ");
  case TensorFormat::kHWC8:
    return std::string("HWC8 ");
  case TensorFormat::kCHW4:
    return std::string("CHW4 ");
  case TensorFormat::kCHW16:
    return std::string("CHW16");
  case TensorFormat::kCHW32:
    return std::string("CHW32");
  case TensorFormat::kHWC:
    return std::string("HWC  ");
  case TensorFormat::kDLA_LINEAR:
    return std::string("DLINE");
  case TensorFormat::kDLA_HWC4:
    return std::string("DHWC4");
  case TensorFormat::kHWC16:
    return std::string("HWC16");
  default:
    return std::string("None ");
  }
}

// Get the string of a TensorRT layer kind
std::string layerTypeToString(LayerType layerType) {
  switch (layerType) {
  case LayerType::kCONVOLUTION:
    return std::string("CONVOLUTION");
  case LayerType::kCAST:
    return std::string("CAST");
  case LayerType::kACTIVATION:
    return std::string("ACTIVATION");
  case LayerType::kPOOLING:
    return std::string("POOLING");
  case LayerType::kLRN:
    return std::string("LRN");
  case LayerType::kSCALE:
    return std::string("SCALE");
  case LayerType::kSOFTMAX:
    return std::string("SOFTMAX");
  case LayerType::kDECONVOLUTION:
    return std::string("DECONVOLUTION");
  case LayerType::kCONCATENATION:
    return std::string("CONCATENATION");
  case LayerType::kELEMENTWISE:
    return std::string("ELEMENTWISE");
  case LayerType::kPLUGIN:
    return std::string("PLUGIN");
  case LayerType::kUNARY:
    return std::string("UNARY");
  case LayerType::kPADDING:
    return std::string("PADDING");
  case LayerType::kSHUFFLE:
    return std::string("SHUFFLE");
  case LayerType::kREDUCE:
    return std::string("REDUCE");
  case LayerType::kTOPK:
    return std::string("TOPK");
  case LayerType::kGATHER:
    return std::string("GATHER");
  case LayerType::kMATRIX_MULTIPLY:
    return std::string("MATRIX_MULTIPLY");
  case LayerType::kRAGGED_SOFTMAX:
    return std::string("RAGGED_SOFTMAX");
  case LayerType::kCONSTANT:
    return std::string("CONSTANT");
  case LayerType::kIDENTITY:
    return std::string("IDENTITY");
  case LayerType::kPLUGIN_V2:
    return std::string("PLUGIN_V2");
  case LayerType::kSLICE:
    return std::string("SLICE");
  case LayerType::kSHAPE:
    return std::string("SHAPE");
  case LayerType::kPARAMETRIC_RELU:
    return std::string("PARAMETRIC_RELU");
  case LayerType::kRESIZE:
    return std::string("RESIZE");
  case LayerType::kTRIP_LIMIT:
    return std::string("TRIP_LIMIT");
  case LayerType::kRECURRENCE:
    return std::string("RECURRENCE");
  case LayerType::kITERATOR:
    return std::string("ITERATOR");
  case LayerType::kLOOP_OUTPUT:
    return std::string("LOOP_OUTPUT");
  case LayerType::kSELECT:
    return std::string("SELECT");
  case LayerType::kFILL:
    return std::string("FILL");
  case LayerType::kQUANTIZE:
    return std::string("QUANTIZE");
  case LayerType::kDEQUANTIZE:
    return std::string("DEQUANTIZE");
  case LayerType::kCONDITION:
    return std::string("CONDITION");
  case LayerType::kCONDITIONAL_INPUT:
    return std::string("CONDITIONAL_INPUT");
  case LayerType::kCONDITIONAL_OUTPUT:
    return std::string("CONDITIONAL_OUTPUT");
  case LayerType::kSCATTER:
    return std::string("SCATTER");
  case LayerType::kEINSUM:
    return std::string("EINSUM");
  case LayerType::kASSERTION:
    return std::string("ASSERTION");
  case LayerType::kONE_HOT:
    return std::string("ONE_HOT");
  case LayerType::kNON_ZERO:
    return std::string("NON_ZERO");
  case LayerType::kGRID_SAMPLE:
    return std::string("GRID_SAMPLE");
  case LayerType::kNMS:
    return std::string("NMS");
  case LayerType::kREVERSE_SEQUENCE:
    return std::string("REVERSE_SEQUENCE");
  case LayerType::kNORMALIZATION:
    return std::string("NORMALIZATION");
  case LayerType::kPLUGIN_V3:
    return std::string("PLUGIN_V3");
  default:
    return std::string("Unknown");
  }
}

// Get the string of a TensorRT tensor location
std::string locationToString(TensorLocation location) {
  switch (location) {
  case TensorLocation::kHOST:
    return std::string("HOST");
  case TensorLocation::kDEVICE:
    return std::string("DEVICE");
  default:
    return std::string("None");
  }
}

std::string ioModeToString(TensorIOMode iomode) {
  switch (iomode) {
  case TensorIOMode::kINPUT:
    return std::string("INPUT");
  case TensorIOMode::kOUTPUT:
    return std::string("OUTPUT");
  default:
    return std::string("None");
  }
}

} // namespace inference