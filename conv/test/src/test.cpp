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

  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONV);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVSHARED);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCONST);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVSHAREDTILE);
  //auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCONSTTILE);
  auto blur = blurImage(grayValues.data(), filter.data(), img.rows, img.cols, ConvType::CONVCACHE);
  
//  for (int i = 0; i < img.rows; ++i) {
//    for (int j = 0; j < img.cols; ++j) {
//      printf("%d ", blur[i*img.cols + j]);
//    }
//    printf("\n");
//  }
  
  // copy to image
  Mat blurMat(img.rows, img.cols, CV_8UC1, Scalar(0));
  for (int rowId = 0; rowId < img.rows; ++rowId) {
      for (int colId = 0; colId < img.cols; ++colId) {
        int offset = rowId * img.cols + colId;
        blurMat.at<uchar>(rowId, colId) = blur[offset];
      }
  }

  imwrite("outgoat.jpg", blurMat);
  

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
  
  switch (algr) {
    case ConvType::CONV:
      conv(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVSHARED:
      conv_shared(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVCONST:
      conv_constant(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVSHAREDTILE:
      conv_shared_tile(d_gray, d_filter, d_blur, RADIUS, height, width);
      break;
    case ConvType::CONVCONSTTILE:
      conv_const_tile(d_gray, d_filter, d_blur, RADIUS, height, width);
    case ConvType::CONVCACHE:
      conv_cache(d_gray, d_filter, d_blur, RADIUS, height, width);
  } 
 
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