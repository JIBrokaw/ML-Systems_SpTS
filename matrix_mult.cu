/*****************************************************************************
 * File:        matrix_mult.cu
 *
 * Run:         ./matrix_mult
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "common_string.h"
#include "kernel.cu"

//Helper function to print a matrix in readable form
void printMatrix(float* matrix, int matrix_size){
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            printf(" %f", matrix[matrix_size*i + j]);
        }
        printf("\n");
    }
}

bool multiply(int matrix_size, float sparsity)
{
    unsigned int bytes = matrix_size * matrix_size * sizeof(float);
    float* h_A, *h_B, *h_out;
    float* d_A, *d_B, *d_out;
    double throughput;

    // allocate host memory
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    // initialize two sparse triangular matrices
    for (int i = 0; i < matrix_size; i++){
        for (int j = 0; j <= i; j++){
            h_A[i*matrix_size + j] = rand()/(float)RAND_MAX < sparsity ? rand() / (float)RAND_MAX : 0;
            h_B[i*matrix_size + j] = rand()/(float)RAND_MAX < sparsity ? rand() / (float)RAND_MAX : 0;
        }
    }
    printf("Matrices initialized!\n");

    // allocate device memory
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    printf("\n %s\n", "Launch Kernel....");

    struct timeval startTime;
    struct timeval endTime;

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(1, 1, 1);
    printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Matrix size: %d*%d\n", matrix_size, matrix_size);

    //todo: warmup
    vanilla_gpu_mult<<<dimBlock,dimGrid>>>(d_A, d_B, d_out, matrix_size);

    printf("\nbasic_mult:\n");
    gettimeofday(&startTime, NULL);
    basic_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    cudaDeviceSynchronize();
    gettimeofday(&endTime, NULL);
    printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    float* gpu_out = (float*)malloc(bytes);
    cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if(matrix_size <= 15){
        printf("\nMatrix A:\n ");
        printMatrix(h_A, matrix_size);
        printf("Matrix B:\n ");
        printMatrix(h_B, matrix_size);
        printf("Result:\n ");
        printMatrix(gpu_out, matrix_size);
    }

    printf("\nvanilla_gpu_mult:\n");
    gettimeofday(&startTime, NULL);
    vanilla_gpu_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    cudaDeviceSynchronize();
    gettimeofday(&endTime, NULL);
    printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if(matrix_size <= 15){
        //printf("\nMatrix A:\n ");
        //printMatrix(h_A, matrix_size);
        //printf("Matrix B:\n ");
        //printMatrix(h_B, matrix_size);
        printf("\nResult:\n ");
        printMatrix(gpu_out, matrix_size);
    }

    // //todo: getting result
    // printf("[Kernel %s] Throughput = %.4f GB/s, Time = %.5f ms\n",
    //     kernel_type, throughput, elapsedTime(timer)*1000);
    // printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // free memory (both device and host mem)

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    free(h_A);
    free(h_B);
    free(h_out);
    // free(cpu_out);
    free(gpu_out);

    return true;
}

/*
 * Argument:
 *      "--width=<N>"       : Specify the number of width of input image (default: 1024)
 *      "--height=<N>"      : Specify the number of height of input image (default: 2048)
 *      "--channel=<N>"     : Specify the number of channels of input image (default: 1, <= 3)
 *      "--filter=<N>"      : Specify the number of filter width for convolution (default: 5)
*/

int main(int argc, char** argv)
{
    //printf("[Multiplying matrices...]\n\n");

    int matrix_size = 10;
    float sparsity = 0.7;

    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        matrix_size = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "sparsity")) {
        sparsity = getCmdLineArgumentInt(argc, (const char **)argv, "sparsity");
    }

    int dev = 0;
    cudaSetDevice(dev);

    multiply(matrix_size, sparsity);

    // bool result;
    // result = compute(width, height, channels, kernel_width, "vanilla");
    // printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    //
    //
    // result = compute(width, height, channels, kernel_width, "shared_mem");
    // printf(result ? "Test PASSED\n" : "Test FAILED!\n");


    //todo: bonus
    //result = compute(width, height, channels, kernel_width, "optimized");
    //printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    cudaDeviceReset();

    return 0;
}
