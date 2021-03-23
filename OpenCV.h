// OpenCV.H
#ifndef __OpenCV_H__
#define __OpenCV_H__
#include <string>
#include "opencv2/opencv.hpp"

class OpenCV{
    int c = 0;
    std::string s;
    std::string img_path;


    public:
    OpenCV(std::string);
    std::string int2str(int &);
    int getFrames();
    void getImage(int *arr);
    int getCols();
    int getRows();
    void mat2arr(cv::Mat img, int *arr);
    cv::Mat arr2mat(int *arr,int w,int h);
    void saveToimg(int *arr,std::string name,int w,int h);
  };


#endif