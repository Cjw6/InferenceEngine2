#pragma once

#include <string>

namespace inference {

class TensorDataPointer;
class TensorData;

int SaveTensorDataToFile(TensorDataPointer *buffer,
                         const std::string &file_path);

TensorData LoadTensorDataFromFile(const std::string &file_path);

} // namespace inference