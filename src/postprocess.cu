#include "postprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"

__global__ void argmax_kernel(const float* input, uint8_t* output, int C, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // height

    if (x >= W || y >= H) return;

    int max_idx = 0;
    float max_val = -1e10;

    for (int c = 0; c < C; ++c)
    {
        // input index: (c, y, x) => c * H * W + y * W + x
        float val = input[c * H * W + y * W + x];
        if (val > max_val)
        {
            max_val = val;
            max_idx = c;
        }
    }

    output[y * W + x] = static_cast<uint8_t>(max_idx);  // 写入类别索引
}


void cuda_postprocess(const float* gpu_input, int output_c, int output_h, int output_w,
    uint8_t* cpu_mask_out)
{
    // 分配 GPU 内存用于 mask（[H, W]）
    uint8_t* gpu_mask = nullptr;
    size_t mask_bytes = output_h * output_w * sizeof(uint8_t);
    cudaMalloc(&gpu_mask, mask_bytes);

    // 设置 CUDA 网格尺寸
    dim3 block(16, 16);
    dim3 grid((output_w + block.x - 1) / block.x, (output_h + block.y - 1) / block.y);

    // 启动 CUDA kernel
    argmax_kernel <<<grid, block>>> (gpu_input, gpu_mask, output_c, output_h, output_w);
    cudaDeviceSynchronize();

    // 拷贝结果回 CPU
    cudaMemcpy(cpu_mask_out, gpu_mask, mask_bytes, cudaMemcpyDeviceToHost);

    // 释放 GPU 中间缓冲区
    cudaFree(gpu_mask);
}

