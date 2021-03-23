#include "OpenCV.h"
#include "OpenCV.cpp"
using namespace std;
void printVEC(int *a,int w);

int main() {
    OpenCV cv2("/tmp/tmp.whjZ5QMQTf/outputs/0.jpg");
    //cv2.getFrames();
    int h = cv2.getRows();
    int w = cv2.getCols();
    cout<<"Ancho: "<<w<<endl;
    cout<<"Alto: "<<h<<endl;
    int arr[h*w];
    cv2.getImage(arr);

    cout<<"Matriz: "<<endl;
    printVEC(arr,w*h);
    cv2.saveToimg(arr,"/tmp/tmp.whjZ5QMQTf/test.jpg",w,h);

    return 0;
}

void printVEC(int *a,int w){
    cout<<"["<<"";
    for(int i = 0; i < w; i++){
        cout<<a[i]<<", ";
    }
    cout<<"]\n"<<endl;
}