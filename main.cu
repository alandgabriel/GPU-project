#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
//#include "OpenCV.h"
//#include "OpenCV.cpp"

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32
#define W 1500
#define H 1500

using namespace std;

void insertionSort_(int *arr, int n);
void swap_(int *a, int *b);
void compareVectors(int *a, int *b);
void generateRandom(int *a,int rows, int cols);
void serial_median_filter3x3();
void parallel_median_filter3x3();
void printMAT(int *a);
void printVEC(int *a,int w);
void copiarValores(int *a, int *b);


//Cargar imagen
//OpenCV cv2("/tmp/tmp.whjZ5QMQTf/outputs/0.jpg");
//int H = cv2.getRows();
//int W = cv2.getCols();

//global variables
int *h_img, *filtered_img_serial, *filtered_img_par;
int *d_img, *d_img_res;
int size = W*H*sizeof(int);

// global timers
double serialTimer = 0.0;
float parallelTimer = 0.0;

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}

#define KERNEL_R 3

__device__ void swap(int *a, int *b){
    int d = *a;
    *a = *b;
    *b = d;
}

__device__ void insertionSort(int *arr, int n){
    for(int i = 1; i < n; i++){
        int j = i - 1;
        int key = arr[i];
        while(j >= 0 && arr[j] > key){
            if(arr[j] > arr[j + 1])
                swap(&arr[j], &arr[j + 1]);
            j--;
        }
    }
}

__device__ void sort(int *a, int *b, int *c) {
    int d;
    if(*a > *b){
        d = *a;
        *a = *b;
        *b  = d;
    }
    if(*a > *c){
        d = *a;
        *a = *c;
        *c  = d;
    }
    if(*b > *c){
        d = *b;
        *b = *c;
        *c  = d;
    }
}


__global__ void medianFilter3x3(const int *src, int w, int h, int *dst){
    const int r = KERNEL_R;
    int rHalf = r / 2;
    int imgBlock[r * r];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int i, j, k, l;
    if(x  > 0 && x  < w-1 && y  > 0 && y < h-1){
        for (i = x - rHalf, k = 0; i <= x + rHalf; i++, k++) {
            for (j = y - rHalf, l = 0; j <= y + rHalf; j++, l++) {
                imgBlock[k* r + l ] = src[i* w + j ];
            }
        }

        insertionSort(imgBlock,r*r);

        //Set median
        dst[x* w + y ] = imgBlock[rHalf + rHalf * r];
    }
    else if(x < w && y < h){
        dst[x*w + y ] = src[x*w + y];
    }
}

int main() {

    // Reservar memoria en host
    h_img = (int *) malloc(size);
    filtered_img_par = (int *) malloc(size);
    filtered_img_serial = (int *) malloc(size);

    //Convertir imagen a arreglo
    //cv2.getImage(h_img);

    //generar img aleatoria
    generateRandom(h_img,W,H);
    //printf("Source image: \n");
    //printMAT(h_img);

    // Reservar memoria en device
    cudaMalloc((void **)&d_img, size);
    // Transferir datos de host a device
    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);

    serial_median_filter3x3();
    parallel_median_filter3x3();
    compareVectors(filtered_img_par,filtered_img_serial);
    cout << "Serial: " << serialTimer << " Parallel: " << parallelTimer / 1000 <<endl;
    cout << "Speed-up: " << serialTimer / (parallelTimer /1000)<< "X"<<endl;
    cout << "\n"<<endl;

    CUDA_CALL(cudaFree(d_img));
    CUDA_CALL(cudaFree(d_img_res));

    //Guardar resultado
    //cv2.saveToimg(filtered_img_serial,"/tmp/tmp.UfSt4NNo2q/filtro.jpg",W,H);

    free(h_img);
    free(filtered_img_par);
    free(filtered_img_serial);

    return 0;
}

void parallel_median_filter3x3(){
    // Reservar memoria en device
    cudaMalloc((void **)&d_img_res, size);
    dim3 blockSize = dim3(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 gridSize = dim3(ceil(W / BLOCKDIM_X)+ 1, ceil(H / BLOCKDIM_Y)+ 1);

    // Definir timers
    cudaEvent_t start, stop;

    // Eventos para tomar tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    medianFilter3x3<<<gridSize, blockSize>>>(d_img, W, H, d_img_res);
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&parallelTimer, start, stop);

    cout<< "Tiempo en generar el filtro en paralelo: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;

    CUDA_CALL(cudaMemcpy(filtered_img_par, d_img_res, size, cudaMemcpyDeviceToHost));

    //printf("Source image: \n");
    //printMAT(h_img);

    //printf("Blurred image par: \n");
    //printMAT(filtered_img_par);
}


void serial_median_filter3x3(){
    clock_t start = clock();
    for(int i=0;i<W;i++){
        for(int j=0;j<H;j++){
            int box3x3[9];
            if(i>0 && j>0 && i<W-1 && j<H-1){
                box3x3[0] = h_img[(i-1)*W+(j-1)];
                box3x3[1] = h_img[(i-1)*W+(j)];
                box3x3[2] = h_img[(i-1)*W+(j+1)];
                box3x3[3] = h_img[(i)*W+(j-1)];
                box3x3[4] = h_img[(i)*W+(j)];
                box3x3[5] = h_img[(i)*W+(j+1)];
                box3x3[6] = h_img[(i+1)*W+(j-1)];
                box3x3[7] = h_img[(i+1)*W+(j)];
                box3x3[8] = h_img[(i+1)*W+(j+1)];

                insertionSort_(box3x3,9);
                int median = box3x3[4];
                filtered_img_serial[i*W+j]=median;
            }
            else{
                filtered_img_serial[i*W+j]=h_img[i*W+j];
            }

        }
    }
    clock_t end = clock();
    serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
    cout << "Tiempo en obtener filtro img serial: " << serialTimer << endl;
    //printf("Blurred image ser: \n");
    //printMAT(filtered_img_serial);
}


void swap_(int *a, int *b){
    int d = *a;
    *a = *b;
    *b = d;
}

void insertionSort_(int *arr, int n){
    for(int i = 1; i < n; i++){
        int j = i - 1;
        int key = arr[i];
        while(j >= 0 && arr[j] > key){
            if(arr[j] > arr[j + 1])
                swap_(&arr[j], &arr[j + 1]);
            j--;
        }
    }
}

void generateRandom(int *a,int rows, int cols){
    // Initialize seed
    srand(time(NULL));
    for(int i=0; i<rows*cols; i++){
        a[i] = rand() % 256;
    }
}
void compareVectors(int *a, int *b){
    cout<<"Total elements "<<W*H<< endl;
    int diff = 0;
    for(int i= 0; i<W*H; i++)
        if(a[i] != b[i]){
            diff++;
        }

    if(diff>0){
        cout<< diff <<" elements different" << endl;
    }
    else
        cout << "Vectors are equal!..." << endl;
}

void printMAT(int *a){
    cout<<"[\n"<<endl;
    for(int i = 0; i < W; i++){
        cout<<"["<<"";
        for(int j = 0; j < H; j++){
            cout<<a[i* W + j]<<" ";
        }
        cout<<"]\n"<<endl;
    }
    cout<<"]\n"<<endl;
}
void printVEC(int *a,int w){
    cout<<"[\n"<<endl;
    for(int i = 0; i < w; i++){
        cout<<a[i]<<" ";
    }
    cout<<"]\n"<<endl;
}

void copiarValores(int *a, int *b){
    for(int i=0;i<W*H;i++){
        b[i]=a[i];
    }
}