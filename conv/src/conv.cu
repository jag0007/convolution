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

__global__ void conv_shared_tile_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
// assume tileDim == blockDim >= FilterDim  

  extern __shared__ float s[];
  float *F_ds = s;                                // shared filter data
  unsigned char *N_ds = (unsigned char *)&F_ds[(2*r+1) * (2*r+1)];     // shared input data

  // __shared__ F_ds[2*RADIUS+1][2*RADIUS+1];
  // __shared__ N_ds[TILE_DIM][TILE_DIM];

  int inputDim = blockDim.x;                      // input tile dim
  int outputDim = inputDim - 2*r;                 // output tile dim
  int filterDim = 2*r+1;

  int inCol = blockIdx.x*outputDim + threadIdx.x - r;
  int inRow = blockIdx.y*outputDim + threadIdx.y - r;

  // load F into shared mem
  if (threadIdx.x < filterDim && threadIdx.y < filterDim) {
    F_ds[threadIdx.y * filterDim + threadIdx.x] = F[(threadIdx.y*filterDim) + threadIdx.x];
  }
  __syncthreads();
  
  // load input into shared
  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = N[inRow*width + inCol];
  } else {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = 0;
  } 
  
  __syncthreads();
  
  // do the conv
  int tileCol = threadIdx.x - r;
  int tileRow = threadIdx.y - r;
  if (tileCol >= 0 && tileCol < outputDim && tileRow >= 0 && tileRow < outputDim) {
    if (inRow >= 0  && inRow < height && inCol >= 0 && inCol < width) {
      float outPix = 0.0f;
      for (int frow = 0; frow < filterDim; ++frow) {
        for (int fcol = 0; fcol < filterDim; ++fcol) {
            outPix += (float) N_ds[(tileRow+frow)*blockDim.x + tileCol+fcol] * F_ds[frow * filterDim + fcol];
        }
      }
      P[inRow * width + inCol] = outPix;  
    }
  }
}

__global__ void conv_const_tile_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
// assume tileDim == blockDim >= FilterDim  

  extern __shared__ unsigned char k[];
  //float *F_ds = s;                                // shared filter data
  unsigned char *N_ds = k;     // shared input data

  // __shared__ F_ds[2*RADIUS+1][2*RADIUS+1];
  // __shared__ N_ds[TILE_DIM][TILE_DIM];

  int inputDim = blockDim.x;                      // input tile dim
  int outputDim = inputDim - 2*r;                 // output tile dim
  int filterDim = 2*r+1;

  int inCol = blockIdx.x*outputDim + threadIdx.x - r;
  int inRow = blockIdx.y*outputDim + threadIdx.y - r;

  // load input into shared
  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = N[inRow*width + inCol];
  } else {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = 0;
  } 
  
  __syncthreads();
  
  // do the conv
  int tileCol = threadIdx.x - r;
  int tileRow = threadIdx.y - r;
  if (tileCol >= 0 && tileCol < outputDim && tileRow >= 0 && tileRow < outputDim) {
    if (inRow >= 0  && inRow < height && inCol >= 0 && inCol < width) {
      float outPix = 0.0f;
      for (int frow = 0; frow < filterDim; ++frow) {
        for (int fcol = 0; fcol < filterDim; ++fcol) {
            outPix += (float) N_ds[(tileRow+frow)*blockDim.x + tileCol+fcol] * cF[frow * filterDim + fcol];
        }
      }
      P[inRow * width + inCol] = outPix;  
    }
  }
}

__global__ void conv_cache_kernel(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {

  extern __shared__ unsigned char l[];
  unsigned char *N_ds = l;
  
  int filterDim = 2*r+1; 
  
  int Col = blockIdx.x*blockDim.x + threadIdx.x;
  int Row = blockIdx.y*blockDim.y + threadIdx.y;

   // load input into shared
  if (Row < height && Col < width) {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = N[Row*width + Col];
  } else {
    N_ds[threadIdx.y*blockDim.x + threadIdx.x] = 0;
  }  
  
  __syncthreads();

  float outPix = 0.0f;
  if (Row < height && Col < width) {
    for (int frow = 0; frow < filterDim; ++frow) {
      for (int fcol = 0; fcol < filterDim; ++fcol) {
        int inRow = threadIdx.y+frow-r;
        int inCol = threadIdx.x+fcol-r;
        ////if (Row == 0 && Col == 4){
          ////printf("frow, fcol: (%d,%d) inRow, inCol: (%d,%d)\n", frow, fcol, inRow, inCol);
        ////}
        if (inRow >=0 && inRow < blockDim.y &&
            inCol >=0 && inCol < blockDim.x) {

              ////if (Row == 0 && Col == 4){
                ////printf("frow, fcol: (%d,%d) N_ds: %d, cF: %f\n", frow, fcol, N_ds[inRow*blockDim.x + inCol],cF[frow*filterDim + fcol]);
              ////}
              outPix += (float) N_ds[inRow*blockDim.x + inCol] * cF[frow*filterDim + fcol];
              ////if (Row == 0 && Col == 4){
                ////printf("outPix: %f\n", outPix);
              ////}
        } else {
          if (Row-r+frow >= 0 && Row-r+frow < height &&
              Col-r+fcol >= 0 && Col-r+fcol < width) {
              //if (Row == 0 && Col == 4){
                //printf("frow, fcol: (%d,%d) N: %d, cF: %f\n", frow, fcol, N[(Row-r+frow)*width + Col-r+fcol], cF[frow*filterDim + fcol]);
              //}
              outPix += (float) N[(Row-r+frow)*width + Col-r+fcol] * cF[frow*filterDim + fcol];
              ////if (Row == 0 && Col == 4){
                ////printf("outPix: %f\n", outPix);
              ////}
            }
        }
      }
    }
    P[Row*width + Col] = (unsigned char) outPix;
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
  int outBlockSize = blockSizeX - 2*r;
  int gridSizeX = (width + outBlockSize -1) / outBlockSize;
  int gridSizeY = (height + outBlockSize -1) / outBlockSize;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  int sharedMemorySizeFilter = (2*r+1) * (2*r+1) * sizeof(float);
  int sharedMemorySizeInputData = blockSizeX*blockSizeY * sizeof(unsigned char);
  conv_shared_tile_kernel<<<gridSize, blockSize, sharedMemorySizeFilter + sharedMemorySizeInputData>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}

void conv_const_tile(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int outBlockSize = blockSizeX - 2*r;
  int gridSizeX = (width + outBlockSize -1) / outBlockSize;
  int gridSizeY = (height + outBlockSize -1) / outBlockSize;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);

  int sharedMemorySizeInputData = blockSizeX*blockSizeY * sizeof(unsigned char);
  conv_const_tile_kernel<<<gridSize, blockSize, sharedMemorySizeInputData>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}

void conv_cache(unsigned char *N, float *F, unsigned char *P, int r, int height, int width) {
  int blockSizeX = 2;
  int blockSizeY = 2;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize(blockSizeX, blockSizeY);
  dim3 gridSize(gridSizeX, gridSizeY);
  printf("blockSizeX: %d, blockSizeY: %d, gridSizeX: %d, gridSizeY: %d\n",blockSizeX, blockSizeY, gridSizeX, gridSizeY);
  int sharedMemorySizeInputData = blockSizeX*blockSizeY * sizeof(unsigned char);
  conv_cache_kernel<<<gridSize, blockSize, sharedMemorySizeInputData>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());

}  