#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "matmul.cu"

constexpr uint64_t kLaunch_BlockSide = 16;
constexpr uint64_t kTime_WarmupRuns = 3;
constexpr uint64_t kTime_MeasuredRuns = 20;
constexpr uint64_t kInput_Seed = 1;
constexpr int kInput_Range = 16;
constexpr float kTime_MicrosecondsPerMillisecond = 1000.0f;
constexpr int kExit_Success = 0;
constexpr int kExit_Failure = 1;
constexpr int kArgument_FirstSize = 2;

class MatmulDriver {
 public:
    /*!
     *  \brief  run the kernel on random n x n int32 matrices and either time it or check it
     *  \param  n           the matrix side
     *  \param  do_check    whether to compare the product with a CPU reference instead of timing
     *  \return kExit_Success when the launch succeeded and, when checking, every element matched
     */
    static int run(uint64_t n, bool do_check);
};

int MatmulDriver::run(uint64_t n, bool do_check) {
    int retval = kExit_Success;
    uint64_t nb_element = n * n;
    uint64_t nb_byte = nb_element * sizeof(int);
    uint64_t i = 0;
    uint64_t k = 0;
    uint64_t row = 0;
    uint64_t column = 0;
    int64_t sum = 0;
    float elapsed_ms = 0.0f;
    int* device_a = nullptr;
    int* device_b = nullptr;
    int* device_c = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<int> list_a(nb_element);
    std::vector<int> list_b(nb_element);
    std::vector<int> list_c(nb_element);
    dim3 block(kLaunch_BlockSide, kLaunch_BlockSide);
    dim3 grid(
        (n + kLaunch_BlockSide - 1) / kLaunch_BlockSide,
        (n + kLaunch_BlockSide - 1) / kLaunch_BlockSide
    );

    srand(kInput_Seed);
    for (i = 0; i < nb_element; ++i) {
        list_a[i] = rand() % kInput_Range;
        list_b[i] = rand() % kInput_Range;
    }
    if (
        cudaMalloc(&device_a, nb_byte) != cudaSuccess
        || cudaMalloc(&device_b, nb_byte) != cudaSuccess
        || cudaMalloc(&device_c, nb_byte) != cudaSuccess
        || cudaEventCreate(&start) != cudaSuccess
        || cudaEventCreate(&stop) != cudaSuccess
    ){
        fprintf(stderr, "n=%lu: cannot allocate on the device\n", n);
        retval = kExit_Failure;
        goto exit;
    }
    cudaMemcpy(device_a, list_a.data(), nb_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, list_b.data(), nb_byte, cudaMemcpyHostToDevice);

    // one launch for the check, warm-up and measured launches for the time
    for (i = 0; i < (do_check ? 1 : kTime_WarmupRuns); ++i) {
        square_matrix_mul<<<grid, block>>>(device_a, device_b, device_c, static_cast<int>(n));
    }
    if (!do_check) {
        cudaEventRecord(start);
        for (i = 0; i < kTime_MeasuredRuns; ++i) {
            square_matrix_mul<<<grid, block>>>(device_a, device_b, device_c, static_cast<int>(n));
        }
        cudaEventRecord(stop);
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "n=%lu: the kernel failed: %s\n", n, cudaGetErrorString(cudaGetLastError()));
        retval = kExit_Failure;
        goto exit;
    }
    if (!do_check) {
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        printf(
            "n=%lu time_us=%.1f\n",
            n,
            elapsed_ms * kTime_MicrosecondsPerMillisecond / kTime_MeasuredRuns
        );
        goto exit;
    }

    cudaMemcpy(list_c.data(), device_c, nb_byte, cudaMemcpyDeviceToHost);
    for (row = 0; row < n; ++row) {
        for (column = 0; column < n; ++column) {
            sum = 0;
            for (k = 0; k < n; ++k) {
                sum += static_cast<int64_t>(list_a[row * n + k]) * list_b[k * n + column];
            }
            if (static_cast<int>(sum) != list_c[row * n + column]) {
                fprintf(
                    stderr,
                    "n=%lu: c[%lu][%lu] is %d, expected %d\n",
                    n,
                    row,
                    column,
                    list_c[row * n + column],
                    static_cast<int>(sum)
                );
                retval = kExit_Failure;
                goto exit;
            }
        }
    }
    printf("n=%lu correct\n", n);

exit:
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    return retval;
}

int main(int argc, char** argv) {
    int retval = kExit_Success;
    int argument_index = 0;
    bool do_check = false;

    if (
        argc <= kArgument_FirstSize
        || (strcmp(argv[1], "time") != 0 && strcmp(argv[1], "check") != 0)
    ){
        fprintf(stderr, "usage: %s time|check <n>...\n", argv[0]);
        retval = kExit_Failure;
        goto exit;
    }
    do_check = strcmp(argv[1], "check") == 0;
    for (argument_index = kArgument_FirstSize; argument_index < argc; ++argument_index) {
        if (MatmulDriver::run(strtoull(argv[argument_index], nullptr, 10), do_check) != kExit_Success) {
            retval = kExit_Failure;
        }
    }

exit:
    return retval;
}
