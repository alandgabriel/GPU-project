// OpenCV.H
#ifndef __OpenCV_H__
#define __OpenCV_H__
#include <string>

class OpenCV{
    int c = 0;
    std::string s;

    public:
    std::string int2str(int &);
    int getFrames();
  };


#endif