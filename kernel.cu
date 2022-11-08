__global__ void basic_mult(float* matrix_A, float* matrix_B, float* out, int size)
{
    for (int output_col = 0; output_col < size; output_col++){
        for (int row = 0; row < size; row++){
            float val = 0;
            for (int col = 0; col <= row; col++){
                val += matrix_A[row*size + col]*matrix_B[col*size + output_col];
            }
            out[row*size + output_col] = val;
        }
    }
}

__global__ void vanilla_gpu_mult(float* matrix_A, float* matrix_B, float* out, int size)
{
    int tx=blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float val=0;
    for(int i=0;i<size;i++){
        val+=matrix_A[ty*size+i]*matrix_B[i*size+tx];
    }
    out[ty*size+tx]=val;
}