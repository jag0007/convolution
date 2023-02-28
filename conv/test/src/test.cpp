#include <cstdio>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <conv.h>
#include <cstdio>
#include <Timer.hpp>
using namespace cv;
using namespace std;

#define RADIUS 2 


Mat getGrayGoat();
std::vector<unsigned char> blurImage(const unsigned char *img, const float *filter, int height, int width, ConvType algr);
void printStatsConv(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void printStatsShared(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void printStatsConstant(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void printStatsSharedTile(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void printStatsConstantTile(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void printStatsCache(double elapsed_ms, int runs, int r, int height, int width, int blockSize);
void writeStats(double, double, double);

struct tmp {
  int rows;
  int cols;
};

Mat getGrayGoat() {

  Mat img = imread("goats/goats.png", IMREAD_COLOR); // this probably needs to change
  
  //imshow("Goat!", img);

  int red = 0;
  int green = 0;
  int blue = 0;
  Mat greyMat(img.rows, img.cols, CV_8UC1, Scalar(0));
  for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
    for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
      auto & vec = img.at<cv::Vec<uchar, 3>>(rowIdx, colIdx);
      blue = vec[0];
      green = vec[1];
      red = vec[2];
      greyMat.at<uchar>(rowIdx, colIdx) = 
        red * 3.0 / 10.0 + green * 6.0 / 10.0 + blue * 1.0 / 10.0;
    }
  }

  return greyMat;
  //imshow("Grayscale goat!", greyMat);
  //imwrite("goats/Grayscalegoat.jpg", greyMat);
  
}

int main(int argc, char** argv) {
 
  // load up gray image 
   Mat img = getGrayGoat();
  //tmp img;
  //img.rows = 5;
  //img.cols = 5;
  // store gray image in column major format
  std::vector<unsigned char> grayValues(img.rows * img.cols, 1);
  for (int rowId = 0; rowId < img.rows; ++rowId) {
    for (int colId = 0; colId < img.cols; ++colId) {
      grayValues[rowId*img.cols + colId] = img.at<uchar>(rowId, colId);
    }
  }

  // create column major filter 5x5, fill with all 1s
  int filterArea = (RADIUS*2 + 1) * (RADIUS*2 + 1);
  std::vector<float> filter(filterArea, 1.0/ (float) filterArea);

  std::vector<ConvType> runs{ConvType::CONV, ConvType::CONVSHARED, ConvType::CONVCONST, ConvType::CONVSHAREDTILE, ConvType::CONVCONSTTILE, ConvType::CONVCACHE};

  for (auto type : runs) {
    auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, type); 
    Mat blurMat(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int rowId = 0; rowId < img.rows; ++rowId) {
      for (int colId = 0; colId < img.cols; ++colId) {
        int offset = rowId * img.cols + colId;
        blurMat.at<uchar>(rowId, colId) = blur[offset];
      }
    }

  imwrite("outgoat.jpg", blurMat);

  }
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONV);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVSHARED);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCONST);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVSHAREDTILE);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCONSTTILE);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCACHE);
  
//  for (int i = 0; i < img.rows; ++i) {
//    for (int j = 0; j < img.cols; ++j) {
//      printf("%d ", blur[i*img.cols + j]);
//    }
//    printf("\n");
//  }
  
  // copy to image
  //Mat blurMat(img.rows, img.cols, CV_8UC1, Scalar(0));
  //for (int rowId = 0; rowId < img.rows; ++rowId) {
  //    for (int colId = 0; colId < img.cols; ++colId) {
  //      int offset = rowId * img.cols + colId;
  //      blurMat.at<uchar>(rowId, colId) = blur[offset];
  //    }
  //}

  //imwrite("outgoat.jpg", blurMat);
  

  return 0;
}

std::vector<unsigned char> blurImage(const unsigned char *img, const float *filter, int height, int width, ConvType algr) {
  // make things
  unsigned char * d_gray = nullptr;
  unsigned char * d_blur = nullptr;
  float * d_filter = nullptr;
  size_t imgSize = height * width * sizeof(unsigned char);
  size_t filterSize = (RADIUS*2+1) * (RADIUS*2+1) * sizeof(float);
  
  
  // allocate things
  checkCudaErrors(
      cudaMalloc(&d_gray, imgSize)
  );
  checkCudaErrors(
      cudaMalloc(&d_blur, imgSize)
  );
  checkCudaErrors(
      cudaMalloc(&d_filter, filterSize)
  );
  
  
  // load it up
  checkCudaErrors(
      cudaMemcpy(d_gray, img, imgSize, cudaMemcpyHostToDevice)
  );


  if (algr == ConvType::CONVCONST || algr == ConvType::CONVCONSTTILE || algr == ConvType::CONVCACHE) {
    checkCudaErrors(
      cudaMemcpyToSymbol(cF, filter, filterSize)
    );
  } else {
    checkCudaErrors(
        cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice)
    );
  }
  
  void (*convAlg)(unsigned char*, float*, unsigned char*, int, int, int);
  void (*printIt)(double, int, int, int, int, int);
  
  switch (algr) {
    case ConvType::CONV:
      convAlg = conv;
      printIt = printStatsConv;
      //conv(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVSHARED:
      convAlg = conv_shared;
      printIt = printStatsShared;
      //conv_shared(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVCONST:
      convAlg = conv_constant;
      printIt = printStatsConstant;
      //conv_constant(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVSHAREDTILE:
      convAlg = conv_shared_tile;
      printIt = printStatsSharedTile;
      //conv_shared_tile(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVCONSTTILE:
      convAlg = conv_const_tile;
      printIt = printStatsConstantTile;
      //conv_const_tile(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVCACHE:
      convAlg = conv_cache;
      printIt = printStatsCache;
      //conv_cache(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
  } 

  // for timer besting
  convAlg(d_gray, d_filter, d_blur, RADIUS, height, width);
 
  int runs = 1000;
  double elapsed_ms = 0.0;
  Timer timer;
  for (int i = 0; i < runs; ++i) {
    timer.start();
    convAlg(d_gray, d_filter, d_blur, RADIUS, height, width); 
    timer.stop();
    elapsed_ms += timer.elapsedTime_ms();
  }

  printIt(elapsed_ms, runs, RADIUS, height, width, 32);
  
  // copy back
  std::vector<unsigned char> blur(height * width, 0);
  checkCudaErrors(
      cudaMemcpy(blur.data(), d_blur, imgSize, cudaMemcpyDeviceToHost)
  ); 

  // free bird
  checkCudaErrors(
      cudaFree(d_gray)
  );
  checkCudaErrors(
      cudaFree(d_blur)
  );
  checkCudaErrors(
      cudaFree(d_filter)
  );

  return blur;
}


void printStatsConv(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {

  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);

  // Each thread reads filterDim^2 from filter and imputs
  // this ignores edge cases
  double floatReads = numberOfThreads * filterDim * filterDim;
  double charReads = numberOfThreads * filterDim * filterDim;
  double numberOfWrites = numberOfThreads;
  double numberOfFlops = numberOfThreads * filterDim * 2.0; 
  double flopRate = numberOfFlops / elapsed_ms / 1.0e3;
  double effectiveBandwidth = (charReads *  sizeof(unsigned char) + 
                               floatReads * sizeof(float) +
                               numberOfWrites) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONV\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth);
}

void printStatsShared(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {
 
  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);

  // Each thread reads filterDim^2 from imputs
  // each block also reads filterDim^2
  // this ignores edge cases
  double floatReads = numberOfThreads * filterDim * filterDim;
  int blocks = ((width - 1 + blockSize) / blockSize) + ((height - 1 + blockSize) / blockSize);
  double charReads = (double) blocks * filterDim * filterDim;
  double numberOfWrites = numberOfThreads;
  double numberOfFlops = numberOfThreads * filterDim * 2.0; 
  double flopRate = numberOfFlops / (elapsed_ms / 1.0e3);
  double effectiveBandwidth = (charReads * sizeof(unsigned char) + 
                               floatReads * sizeof(float) + 
                               numberOfWrites ) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONVSHARED\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth); 
}

void printStatsConstant(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {
 
  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);

  // Each thread reads filterDim^2 from filter and imputs
  // this ignores edge cases
  double floatReads = 0.0f;
  double charReads = numberOfThreads * filterDim * filterDim;
  double numberOfWrites = numberOfThreads;
  double numberOfFlops = numberOfThreads * filterDim * 2.0; 
  double flopRate = numberOfFlops / (elapsed_ms / 1.0e3);
  double effectiveBandwidth = (charReads * sizeof(unsigned char) + 
                               floatReads * sizeof(float) + 
                               numberOfWrites) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONVCONST\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth); 
}

void printStatsSharedTile(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {
 
  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);
  int inputDim = blockSize;
  int outputTile = (blockSize - (2 * r));

  int blocks = ((height - 1 + inputDim) / inputDim) * ((width - 1 + inputDim)/inputDim);

  // Each thread reads filterDim^2 from filter and imputs
  // this ignores edge cases
  double floatReads = (double) blocks * filterDim * filterDim;
  double charReads = blocks * inputDim * inputDim;
  double numberOfWrites = width * height;
  double numberOfFlops = width * height * filterDim * 2.0; 
  double flopRate = numberOfFlops / (elapsed_ms / 1.0e3);
  double effectiveBandwidth = (charReads * sizeof(unsigned char) + 
                               floatReads * sizeof(float) + 
                               numberOfWrites) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONVSHAREDTILE\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth); 
}

void printStatsConstantTile(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {
 
  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);
  int inputDim = blockSize;
  int outputTile = (blockSize - (2 * r));

  int blocks = ((height - 1 + inputDim) / inputDim) * ((width - 1 + inputDim)/inputDim);

  // Each thread reads filterDim^2 from filter and imputs
  // this ignores edge cases
  double floatReads = 0.0f;
  double charReads = (double) blocks * inputDim * inputDim;
  double numberOfWrites = width * height;
  double numberOfFlops = width * height * filterDim * 2.0; 
  double flopRate = numberOfFlops / (elapsed_ms / 1.0e3);
  double effectiveBandwidth = (charReads * sizeof(unsigned char) + 
                               floatReads * sizeof(float) + 
                               numberOfWrites) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONVCONSTTILE\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth); 
}

void printStatsCache(double elapsedTime_ms, int runs, int r, int height, int width, int blockSize) {
 
  double elapsed_ms = elapsedTime_ms / (double) runs;
  double numberOfThreads = (double) height * (double) width;
  double filterDim = (2.0 * r + 1.0) * (2.0 * r + 1.0);
  int inputDim = blockSize;
  int outputTile = (blockSize - (2 * r));

  int blocks = ((height - 1 + inputDim) / inputDim) * ((width - 1 + inputDim)/inputDim);

  // Each thread reads filterDim^2 from filter and imputs
  // this ignores edge cases
  double floatReads = 0.0f;
  double charReads = (double) blocks * inputDim * inputDim;
  double numberOfWrites = width * height;
  double numberOfFlops = width * height * filterDim * 2.0; 
  double flopRate = numberOfFlops / (elapsed_ms / 1.0e3);
  double effectiveBandwidth = (charReads * sizeof(unsigned char) + 
                               floatReads * sizeof(float) + 
                               numberOfWrites) * 8.0 /
                              (elapsed_ms / 1.0e3);
                          
  printf("CONVCACHE\n\n");
  writeStats(elapsed_ms, flopRate, effectiveBandwidth); 
}

void writeStats(double elapsed_ms, double flopRate, double effectiveBandwidth) {
  printf("Average run time: %lf\n", elapsed_ms);

  printf (
   "\t- Computational Rate:         %20.16e Gflops\n",
   flopRate / 1e9
  );
  printf (
   "\t- Effective Bandwidth:        %20.16e Gbps\n\n\n",
   effectiveBandwidth / 1e9
  );
}