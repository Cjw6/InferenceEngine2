#include "inference/tensor/tensor.h"
#include "inference/tensor/tensor_helper.h"
#include "inference/utils/log.h"

using namespace inference;

int main(int argc, char *argv[]) {
  LogInit();
  std::string src_npy_data = "experiment/npy_data/random_data.npy";
  auto tensor_data = LoadTensorDataFromFile(src_npy_data);
  std::string dst_npy_data = "experiment/npy_data/random_data_loaded.npy";
  SaveTensorDataToFile(&tensor_data.pointer, dst_npy_data);
}
