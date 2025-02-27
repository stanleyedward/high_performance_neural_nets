#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <string>

#define CUDA_CHECK(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                          \
    {                                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                              \
      cudaDeviceReset();                                               \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)


class Timer
{
public:
  Timer(std::string in_name) : name(in_name)
  {
    start_time = std::chrono::system_clock::now();
  }
  ~Timer()
  {
    std::cout << name << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() << " ms" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start_time;
  std::string name;
};

template <const int INPUT_SIZE>
void read_mnist(const std::string &filename, int length, float *x, float *y)
{
  const int labels = 10;
  std::ifstream fin(filename);
  if (!fin.is_open())
  {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  // skip the header row
  std::string header;
  std::getline(fin, header);

  std::string row;
  for (int i = 0; i < length; i++)
  {
    if (!std::getline(fin, row))
    {
      throw std::runtime_error("Not enough rows in the file");
    }
    std::istringstream line_stream(row);
    std::string value;
    // parse label
    if (!std::getline(line_stream, value, ','))
    {
      throw std::runtime_error("Cannot read label");
    }
    int label = std::stoi(value);
    // init one-hot encoded label
    for (int j = 0; j < labels; j++)
    {
      y[labels * i + j] = (j == label) ? 1.0f : 0.0f;
    }
    // parse input features
    for (int j = 0; j < INPUT_SIZE; j++)
    {
      if (!std::getline(line_stream, value, ','))
      {
        throw std::runtime_error("Not enough features in row");
      }
      x[i * INPUT_SIZE + j] = std::stof(value) / 255.0f;
    }
  }
}

