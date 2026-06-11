# Compilers
CC      = gcc
CXX     = g++       # 必须定义 CXX
NVCC    = nvcc

# CUDA 路径
CUDA_PATH = /usr/local/cuda-12.1
CUDA_LIB  = $(CUDA_PATH)/lib64
CUDA_INC  = $(CUDA_PATH)/include

# 编译选项
CFLAGS   = -O3 -g -std=c++11 -D_GNU_SOURCE -D CISS_DEBUG  # 使用 C++11
NVCCFLAGS = -O2 -g -arch=sm_70 --std=c++11 --compiler-options="-fPIC"


# 链接选项
LDFLAGS = -L$(CUDA_LIB) -lcudart -lcuda -lcusolver -lstdc++ -lm

# 目标文件
TARGET = tensor_sgd
OBJECTS = main.o tensor_SGD.o block.o sgd_Kernel.o

all: $(TARGET)

# 关键修改：使用 $(CXX) 而非 $(CC) 链接
$(TARGET): $(OBJECTS)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 编译 .cpp 文件
%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(CUDA_INC) -c $< -o $@

# 编译 .cu 文件
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean