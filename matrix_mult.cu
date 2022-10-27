/*****************************************************************************
 * File:        matrix_mult.cu
 *
 * Run:         ./conv2D
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// __global__
void basic_mult(float* matrix_A, float* matrix_B, float* out, int size)
{
  for (int output_col = 0; output_col < size; output_col++){
    for (int row = 0; row < size; row++){
      float val = 0;
      for (int col = 0; col <= row; col++){
        val += matrix_A[row*size + col]*matrix_B[col*size + output_col];
        printf("Row %d, Column %d: %f\n", row, col, val);
      }
      out[row*size + output_col] = val;
    }
  }
}


bool multiply(int matrix_size)
{
    unsigned int bytes = matrix_size * matrix_size * sizeof(float);
    float* h_A, *h_B, *h_out;
    float* d_A, *d_B, *d_out;
    double throughput;

    // allocate host memory
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    // init inputs
    for (int i = 0; i < matrix_size; i++){
        for (int j = 0; j <= i; j++){
            h_A[i*matrix_size + j] = rand() / (float)RAND_MAX;
            h_B[i*matrix_size + j] = rand() / (float)RAND_MAX;
        }
        for (int j = i+1; j<= matrix_size; j++){
            h_A[i*matrix_size + j] = 0;
            h_B[i*matrix_size + j] = 0;
        }
    }

    basic_mult(h_A, h_B, h_out, matrix_size);

    printf("%f, %f, %f", h_A[0], h_B[0], h_out[0]);


    // // allocate device memory
    // CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    // CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    // CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    // CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpyToSymbol(M, h_kernel, kernel_width*kernel_width*sizeof(float)));

    // launch Kernel
    // printf("\n %s\n", "Launch Kernel....");
    // Timer timer;
    // if (kernel_type == "vanilla") {
    //     // basic 2d conv
    //     //todo: define block and grid size
    //     dim3 dimBlock(16, 16, 1);
	  //     dim3 dimGrid(1, 1, 1);
    //     printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    //     printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //
    //     //todo: warmup
    //
    //     startTime(&timer);
    //     //todo: launch kernel on device and test performance, get results
    //     convolution2D<<<dimBlock, dimGrid>>>(d_in, d_out, width, height, channels, kernel_width);
	  //     cudaDeviceSynchronize();
    //     stopTime(&timer);
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }
    // else if (kernel_type == "shared_mem") {
    //     //  2d conv on shared mem
    //     //todo: define block and grid size
    //
    //     dim3 dimBlock(16, 16, 1);
    //     dim3 dimGrid(16, 16, 1);
    //
    //     printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    //     printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //
    //     //todo: warmup
    //     convolution2D_sharedmem<<<dimBlock, dimGrid>>>(d_in, d_out, width, height, channels, kernel_width);
    //
    //     startTime(&timer);
    //     convolution2D_sharedmem<<<dimBlock, dimGrid>>>(d_in, d_out, width, height, channels, kernel_width);
    //     cudaDeviceSynchronize();
    //     stopTime(&timer);
    //
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }
    // todo: bonus optimization
    /*
    else if (kernel_type == "optimized") {
    }
    */

    // result in CPU
    // float* cpu_out = (float*)malloc(bytes);
    // printf("\nCalculating in CPU...\n");
    // verify(h_in, cpu_out, h_kernel, width, height, channels, kernel_width);
    //
    // int precision = 8;
    // double threshold = 1e-8 * channels*width*height;
    // double diff = 0.0;
    //
    // //todo: compare kernel result with CPU result
    // float* gpu_out = (float*)malloc(bytes);
    // CUDA_CHECK(cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost));
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    //
    // for(int i = 0; i < bytes/sizeof(float); i++){
    //   diff += abs(cpu_out[i] - gpu_out[i]);
    // }
    //
    // printf("Kernel width is %d \n", kernel_width);
    //
    // //todo: getting result
    // printf("[Kernel %s] Throughput = %.4f GB/s, Time = %.5f ms\n",
    //     kernel_type, throughput, elapsedTime(timer)*1000);
    // printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // todo:
    // free memory (both device and host mem)

    // cudaFree(d_out);
    // cudaFree(d_in);
    free(h_A);
    free(h_B);
    free(h_out);
    // free(cpu_out);
    // free(gpu_out);

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
    printf("[Multiplying matrices...]\n\n");

    int matrix_size = 5;

    // if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
    //     width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    // }
    // if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
    //     height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    // }
    // if (checkCmdLineFlag(argc, (const char **)argv, "channels")) {
    //     channels = getCmdLineArgumentInt(argc, (const char **)argv, "channels");
    // }
    // if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
    //     kernel_width = getCmdLineArgumentInt(argc, (const char **)argv, "filter");
    // }

    int dev = 0;
    cudaSetDevice(dev);

    multiply(matrix_size);

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
