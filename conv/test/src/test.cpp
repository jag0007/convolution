#include <cstdio>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

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
  
  // store gray image in column major format
  std::vector<unsigned char> grayValues(img.rows*img.cols, 0);
  for (int rowId = 0; rowId < img.rows; ++rowId) {
    for (int colId = 0; colId < img.cols; ++colId) {
      grayValues[rowId*img.cols + colId];
    }
  }

  // create column major filter 5x5, fill with all 1s
  std::vector<unsigned char> filter((RAIUS*2+1) * (RADIUS*2+1), 1);
  
  // make things
  unsigned char * d_gray = nullptr;
  unsigned char * d_blur = nullptr;
  float * d_filter = nullptr;
  size_t imgSize = img.rows * img.cols * sizeof(unsigned char);
  size_t filterSize = (r*2+1) * (r*2+1) * sizeof(float);
  
  
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
      cudaMemcpy(d_gray, grayValues.data(), imgSize, cudaMemcpyHostToDevice)
  );
  checkCudaErrors(
      cudaMemcpy(d_filter, filter.data(), filterSize, cudaMemcpyHostToDevice)
  );
  
  
  // make call
  conv(d_gray, d_filter, d_blur, img.rows, img.cols, r);
  
  
  // copy back
  std::vector<unsigned char> blur;
  blur.reserve(img.rows * img.cols * sizeof(unsigned char));
  
  
  checkCudaErrors(
      cudaMemcpy(blur.data(), d_blur, imgSize, cudaMemcpyDeviceToHost)
  );
  
  
  // copy to image
  Mat blurMat(img.rows, img.cols, CV_8UC1, Scalar(0));
  for (int rowId = 0; rowId < img.rows; ++rowId) {
      for (int colId = 0; colId < img.cols; ++colId) {
        int offset = rowId * img.cols + colId;
        blurMat.at<uchar>(rowId, colId) = blur[offset];
      }
  }
  

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


  // extract image to vector
  std::vector<unsigned char> gray(img.rows * img.cols);
  for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx){
    for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
      gray[rowIdx*img.cols + colIdx] = img.at<uchar>(rowIdx, colIdx); 
    }
  }  

   

  return 0;
}
