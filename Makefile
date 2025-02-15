#compiler and flags
NVCC      = nvcc
NVCCFLAGS = -I./src -O3 -arch=sm_89  # Update compute capability
LDFLAGS   = -L/usr/local/cuda/lib64
LDLIBS    = -lcudart

# targs and src files
TARGETS   = nn mm activation test
SOURCES   = $(TARGETS:=.cu)
OBJECTS   = $(SOURCES:.cu=.o)

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : %.o
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o
