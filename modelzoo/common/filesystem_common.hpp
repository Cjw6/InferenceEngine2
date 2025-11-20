#pragma once

#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/to_string.h>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace cpptoolkit {

/*
Get all files with the given extension in the given directory.

@param path The path to the directory.
@param ext The extension of the files to get.
@return A vector of paths to the files with the given extension.

example:
auto files = GetAllFilesWithExt("/path/to/dir", ".txt");
*/
inline std::vector<fs::path> GetAllFilesWithExt(const char *path,
                                                std::string ext) {
  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      if (entry.path().extension() == ext) {
        files.push_back(entry.path());
      }
    }
  }
  return files;
}

inline std::vector<std::string> GetImgDataPaths(const std::string &img_path,
                                                const std::string &ext) {
  std::vector<std::string> img_paths;
  if (fs::is_regular_file(img_path)) {
    if (fs::path(img_path).extension() == ext) {
      img_paths.push_back(img_path);
    }
  }
  if (fs::is_directory(img_path)) {
    auto files = cpptoolkit::GetAllFilesWithExt(img_path.c_str(), ".jpg");
    for (auto &file : files) {
      img_paths.push_back(file);
    }
  }
  return img_paths;
}

} // namespace cpptoolkit