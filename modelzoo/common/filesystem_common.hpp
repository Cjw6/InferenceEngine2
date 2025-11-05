#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace cpputils {

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

}  // namespace cpputils