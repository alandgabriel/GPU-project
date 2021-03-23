//
// Created by Ivan Casasola on 21/03/21.
//
// OpenCV.CPP
#ifndef __OpenCV_CPP__
#define __OpenCV_CPP__

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include "OpenCV.h"

using namespace cv;
using namespace std;


OpenCV::OpenCV(string p) {
    img_path = p;
}

string OpenCV::int2str(int &i) {
    string s;
    stringstream ss(s);
    ss << i;
    return ss.str();
}

void OpenCV::getImage(int *arr) {
    string path = samples::findFile(img_path);
    Mat img=imread(path, IMREAD_GRAYSCALE);
    mat2arr(img,arr);
}

int OpenCV::getCols() {
    string path = samples::findFile(img_path);
    Mat img=imread(path, IMREAD_GRAYSCALE);
    int w = img.cols;
    return w;
}
int OpenCV::getRows() {
    string path = samples::findFile(img_path);
    Mat img=imread(path, IMREAD_GRAYSCALE);
    int h = img.rows;
    return h;
}

void OpenCV::mat2arr(Mat img,int *arr){
    uint h = img.rows;
    uint w = img.cols;
    for(int i=0;i<w;i++){
        for(int j=0;j<h;j++){
            uchar uxy=img.at<uchar>(j,i);
            int color = (int) uxy;
            arr[i*w+j]=color;
        }
    }
}

Mat OpenCV::arr2mat(int *arr,int w,int h){
    Mat tmp(w, h, CV_8UC1);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int value = arr[x +  w*y];
            tmp.at<int>(y, x) = value;
        }
    }
    return tmp;
}

void OpenCV::saveToimg(int *arr,string name,int w,int h){
    imwrite(name, Mat(w,h,CV_8U,&arr));
}

int OpenCV::getFrames(){
    VideoCapture cap("/tmp/tmp.whjZ5QMQTf/echo.avi"); // video
    if (!cap.isOpened())
    {
        cout << "No se puede abrir el archivo" << endl;
        return -1;
    }
    //Se obtienen los cuadros por segundo
    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Cuadros por segundo : " << fps << endl;
    //Se obtiene el primer cuadro
    Mat frame;
    bool success = cap.read(frame);
    s = int2str(c);
    while(1){
        cout << "Guardando cuadro "+ s << endl;
        imwrite( "/tmp/tmp.whjZ5QMQTf/outputs/"+s+".jpg", frame);
        c++;
        s = int2str(c);
        Mat frame;
        bool success = cap.read(frame);
        if (!success){
            break;
        }
    }
    cout << "Cuadros guardados" << endl;
    return 0;
}
#endif