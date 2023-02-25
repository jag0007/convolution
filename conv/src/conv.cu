#include <helper_cuda.h>
#include <cstdio>
#include <conv.h>

__global__ void conv_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int outCol=blockIdx.x*blockDim.x + threadIdx.x;
  int outRow=blockIdx.y*blockDim.y + threadIdx.y;

  float outPix = 0.0f;
  for (int frow=0; frow < 2*r+1; frow++) {
    for (int fcol=0; fcol<2*r+1; fcol++) {
      int inRow = outRow + frow - r;
      int inCol = outCol + fcol - r;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) 
        outPix += ((float) N[inRow*width + inCol] * F[frow*(r*2+1) + fcol]);  
    }
  }
  if (outCol >= 0 && outCol < width && outRow >= 0 && outRow < height)
    P[outRow*width + outCol] = (unsigned char) outPix;
}

__global__ void conv_shared_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
// assume tileDim == blockDim >= FilterDim  
  //__shared__ F_ds[2*RADIUS+1][2*RADIUS+1];
  extern __shared__ float F_ds[];

  int outCol = blockIdx.x*blockDim.x + threadIdx.x;
  int outRow = blockIdx.y*blockDim.y + threadIdx.y;

  // load F into shared mem
  if (threadIdx.x < 2*r + 1 && threadIdx.y < 2*r + 1) {
    F_ds[threadIdx.y*(2*r+1) + threadIdx.x] = F[(threadIdx.y*(2*r+1)) + threadIdx.x];
  }
  __syncthreads();

  float outPix = 0.0f;
  for (int frow=0; frow < 2*r+1; frow++) {
    for (int fcol=0; fcol<2*r+1; fcol++) {
      int inRow = outRow + frow - r;
      int inCol = outCol + fcol - r;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) 
        outPix += ((float) N[inRow*width + inCol] * F_ds[frow*(r*2+1) + fcol]);  
    }
  }
  if (outCol >= 0 && outCol < width && outRow >= 0 && outRow < height)
    P[outRow*width + outCol] = (unsigned char) outPix;   
}

__global__ void conv_constant_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int outCol=blockIdx.x*blockDim.x + threadIdx.x;
  int outRow=blockIdx.y*blockDim.y + threadIdx.y;

  float outPix = 0.0f;
  for (int frow=0; frow < 2*r+1; frow++) {
    for (int fcol=0; fcol<2*r+1; fcol++) {
      int inRow = outRow + frow - r;
      int inCol = outCol + fcol - r;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) 
        outPix += ((float) N[inRow*width + inCol] * cF[frow*(r*2+1) + fcol]);  
    }
  }
  if (outCol >= 0 && outCol < width && outRow >= 0 && outRow < height)
    P[outRow*width + outCol] = (unsigned char) outPix;
}

__global__ void conv_shared_tile_kernel(unsigned char *N, float *F, unsigned char *P, int r, int width, int height) {
// assume tileDim == blockDim >= FilterDim  
  extern __shared__ float s[];
  float *F_ds = s;                                // shared filter data
  unsigned char *N_ds = (unsigned char *)&F_ds[(2*r+1) * (2*r+1)];     // shared input data

  // __shared__ F_ds[2*RADIUS+1][2*RADIUS+1];
  // __shared__ N_ds[TILE_DIM][TILE_DIM];

  int inputDim = blockDim.x;                      // input tile dim
  int outputDim = inputDim - 2*r;                 // output tile dim

  int inCol = blockIdx.x*outputDim + threadIdx.x - r;
  int inRow = blockIdx.y*outputDim + threadIdx.y - r;

  // load F into shared mem
  if (threadIdx.x < 2*r + 1 && threadIdx.y < 2*r + 1) {
    F_ds[threadIdx.y * 2*r+1 + threadIdx.x] = F[(threadIdx.y*2*r+1) + threadIdx.x];
  }

  // load input into shared
  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
    N_ds[threadIdx.y*width + threadIdx.x] = N[inRow*width + inCol];
  } else {
    N_ds[threadIdx.y*width + threadIdx.x] = 0.0f;
  }
  __syncthreads();

  // do the conv
  int tileCol = threadIdx.x - r;
  int tileRow = threadIdx.y - r;
  if (tileCol >= 0 && tileCol < outputDim && tileRow >= 0 && tileRow < outputDim) {
    if (inRow >= 0  && inRow < height && inCol >= 0 && inCol < width) {
      float outPix = 0.0f;
      for (int frow = 0; frow < 2*r+1; ++frow) {
        for (int fcol = 0; fcol < r*2 + 1; ++fcol) {
          outPix += N[inRow*width + inCol] * F_ds[frow * 2*r+1 + fcol];
        }
      }
      P[inRow * width + inCol] = outPix;  
    }
  }
}


void conv(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  conv_kernel<<<gridSize, blockSize>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}

void conv_shared(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  int sharedMemorySize = (2*r+1) * (2*r+1) * sizeof(float);
  conv_shared_kernel<<<gridSize, blockSize, sharedMemorySize>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}

void conv_constant(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  int sharedMemorySize = (2*r+1) * (2*r+1) * sizeof(float);
  conv_constant_kernel<<<gridSize, blockSize, sharedMemorySize>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}

void conv_shared_tile(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  int sharedMemorySizeFilter = (2*r+1) * (2*r+1) * sizeof(float);
  int sharedMemorySizeInputData = (width + 2*r) * (height + 2*r) * sizeof(unsigned char);
  conv_constant_kernel<<<gridSize, blockSize, sharedMemorySizeFilter + sharedMemorySizeInputData>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}