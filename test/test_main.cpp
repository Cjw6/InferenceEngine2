#include <cpptoolkit/log/log.h>
#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  LogInit();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}