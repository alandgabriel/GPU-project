#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

class OpenCV{
    int c = 0;
    string s;

    public:
    string int2str(int &);
    int getFrames();
  };

string OpenCV::int2str(int &i) {
    string s;
    stringstream ss(s);
    ss << i;
    return ss.str();
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

int main() {
    OpenCV cv2;
    cv2.getFrames();
    return 0;
}
