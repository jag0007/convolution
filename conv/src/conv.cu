__global__ void conv_kernel(float *N, float *F, float *P, int r, int height, int width) {
  int outCol=blockIdx.x*blockDim.x + threadIdx.x;
  int outRow=blockIdx.y*blockDim.y + threadIdx.y;

  float outPix = 0.0f;
  for (frow=0; frow < 2*r+1; frow++) {
    for (fcol=0; fcol<2*r+1; fcol++) {
      inRow = outRow + frow - r;
      inCol = outCol + fcol - r;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
        outPix += N[inRow*width + inCol] * F[frow*(r*2+1) + fcol];  
    }
  }
  P[outRow*width + outCol];
}

void conv(float *N, float*F, float *P, int r, int height, int width) {
  int blockSizeX = 32;
  int blockSizeY = 32;
  int gridSizeX = (width + blockSizeX -1) / blockSizeX;
  int gridSizeY = (height + blockSizeY -1) / blockSizeY;

  dim3 blockSize = (blockSizeX, blockSizeY);
  dim3 blockSize = (gridSizeX, gridSizeY);

  conv_kernel<<<gridSize, blockSize>>>(N, F, P, r, height, width);

  checkCudaErrors(cudaGetLastError());
  
}
