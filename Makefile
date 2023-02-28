CC=gcc
CXX=g++
NVCC=nvcc

CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Isaxpy/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules

all: build/lib/libTimer.so build/lib/libconv.so build/bin/conv_test

build/lib/libTimer.so: modules Timer/src/Timer.cpp
		@mkdir -p build/.objects/Timer
		$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
			-I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc \
			-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
		@mkdir -p build/lib
		$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
			-L$(CUDAPATH)/lib64
		@mkdir -p build/include
		@ln -sf ../../Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libconv.so: modules conv/src/conv.cu
	@mkdir -p build/.objects/conv
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Iconv/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include \
		-dc -o build/.objects/conv/conv.o \
		conv/src/conv.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/conv/conv-dlink.o build/.objects/conv/conv.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libconv.so build/.objects/conv/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../conv/include/conv.h build/include/conv.h


build/bin/conv_test: build/lib/libconv.so conv/test/src/test.cpp
	@mkdir -p build/bin
	@cp goats/goats.png build/bin/
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-Iconv/include -Iopencv-install/include -Lopencv-install/lib64 \
		-o build/bin/conv_test conv/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib -Wl,-rpath=$(PWD)/opencv-install/lib64\
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lconv -lcudart -lopencv_core -lopencv_imgcodecs \
		-lopencv_highgui -lopencv_imgproc -lTimer

run: build/bin/conv_test
	@rm -f *.nsys-rep runTestsshGPU.i* runTestsshGPU.o* core.*
	@echo -ne "gpu\n1\n\n10gb\n1\nampere\conv_test\n" | \
	run_gpu .runTests.sh > /dev/null
	@sleep 5
	@tail -f runTestsshGPU.o*

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f conv_test.*
