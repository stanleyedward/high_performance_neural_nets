#compiler and flags
NVCC      := nvcc
NVCCFLAGS := -I./src -O3 -arch=sm_89  # mod compute capability to ur gpu
LDFLAGS   := -L/usr/local/cuda/lib64
LDLIBS    := -lcudart

# targs and srcs
TARGETS              := nn mm activation
ACTIVATION_SRCS      := activation.cu src/activation_runner.cu
MM_SRCS              := mm.cu src/mm_runner.cu
NN_SRCS              := nn.cu src/activation_runner.cu src/mm_runner.cu

.PHONY: all clean

all: $(TARGETS)

activation:
	@echo "[INFO] building activation"
	$(NVCC) $(ACTIVATION_SRCS) $(NVCCFLAGS) -o activation.o

mm:
	@echo "[INFO] building matmul"
	$(NVCC) $(MM_SRCS) $(NVCCFLAGS) -o mm.o

nn:
	@echo "[INFO] building nn"
	$(NVCC) $(NN_SRCS) $(NVCCFLAGS) -o nn.o

clean:
	rm -f *.o

