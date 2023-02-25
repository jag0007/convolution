#include <helper_cuda.h>
#include <cstdio>

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
