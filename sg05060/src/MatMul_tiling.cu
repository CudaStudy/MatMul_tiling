#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A = m x k 
// B = k x n
// C = m x n

#define m 8192
#define n 8192
#define k 8192
#define BLOCK_SIZE 16

__global__ void MatMul_tiling(float *_a, float *_b, float *_c, int _m, int _n, int _k) {
    
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    
    if( row >= _m || col >= _n)
        return;
    
    float ret = 0;
    float lhs = 0;
    float rhs = 0;
    //__shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
    //__shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
    int localRow = threadIdx.x;
    int localCol = threadIdx.y;


    for (int bID = 0; bID < ceil((float)_k / BLOCK_SIZE); bID++) {
        int block_offset = bID * BLOCK_SIZE;

        for(int i = 0; i < BLOCK_SIZE; i++) {
            if(row >= _m || (block_offset + localCol) >= _k)
                lhs = 0;
            else 
                lhs = _a[(row * _k) + block_offset + i];

            if(col >= _n || (block_offset + localRow) >= _k)
                rhs = 0;
            else
                rhs = _b[(block_offset + i)*_n + col];
            
            ret += __fmul_rn(lhs, rhs);
            //printf("ret : %0.2f\n",ret);
        }
    }
    _c[(_n * row + col)] = ret;
    return;

}

int main(int argc, char** argv) {
    DS_timer timer(3);
    timer.setTimerName(0, (char *)"[GPU]");
    timer.setTimerName(1, (char *)"[DATA Transfer] : Host->Device");
    timer.setTimerName(2, (char *)"[DATA Transfer] : Device->Host");

    int size_A = m * k;
    int size_B = k * n;
    int size_C = m * n;

    float *a, *b, *c;
    float *_a, *_b, *_c;

    a = new float[size_A]; memset(a,0,sizeof(float)*size_A);
    b = new float[size_B]; memset(b,0,sizeof(float)*size_B);
    c = new float[size_C]; memset(c,0,sizeof(float)*size_C);

    for (int i = 0; i < size_A; i++) {
        a[i] = rand() % 10 + ((rand() % 100) / 100.0);
    }
    for (int i = 0; i < size_B; i++) {
        b[i] = rand() % 10 + ((rand() % 100) / 100.0);
    }

    cudaMalloc(&_a, sizeof(float)*size_A);
    cudaMalloc(&_b, sizeof(float)*size_B);
    cudaMalloc(&_c, sizeof(float)*size_C);

    timer.onTimer(1);
    cudaMemcpy(_a, a, sizeof(float)*size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, sizeof(float)*size_B, cudaMemcpyHostToDevice);
    timer.offTimer(1);
    //printf("1. Host->Devive : A[%d,%d] = %d bytes, B[%d,%d] = %d bytes\n",m,k,m*k*8,k,n,k*n*8);

    dim3 gridDim (ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim (BLOCK_SIZE, BLOCK_SIZE);

    timer.onTimer(0);
    MatMul_tiling<<<gridDim, blockDim>>>(_a,_b,_c,m,n,k);
    cudaDeviceSynchronize();
    timer.offTimer(0);
    printf("2. GPU Matrix Multiplication\n");

    timer.onTimer(2);
    cudaMemcpy(c, _c, sizeof(float)*size_C, cudaMemcpyDeviceToHost);
    timer.offTimer(2);
    //printf("3. Host->Devive : C = [%d*%d] = %dbytes\n",m,n,m*n*8);

    //bool result = true;
    printf("Checking result....");
    //for (int i = 0; i < m; i++) {
    //    for(int j = 0; j < n; j++) {
    //        float ret = 0;
    //        for(int l = 0; l < k; l++)
    //            ret += a[(k*i)+l] * b[(l*n) + j];
    //        if(ret != c[i*n + j]) {
    //            printf("the result %d is not matched! (%0.2f, %0.2f)\n"
    //            , i, ret, c[i*n + j]);
    //            result = false;
    //        }
    //    }
    //}
    //if(result)
    //    printf("kernel works well!\n");
    
    timer.printTimer();
    return 0;
}