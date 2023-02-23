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
  
  // extract image to vector
  std::vector<unsigned char> gray(img.rows * img.cols);
  for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx){
    for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
      gray[rowIdx*img.cols + colIdx] = img.at<uchar>(rowIdx, colIdx); 
    }
  }  

   

  return 0;
}
