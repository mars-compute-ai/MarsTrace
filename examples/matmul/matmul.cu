extern "C" __global__ void square_matrix_mul(const int* a, const int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || column >= n) {
        return;
    }
    int sum = 0;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + column];
    }
    c[row * n + column] = sum;
}
