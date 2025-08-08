#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// void argmax_kernel(const float* input, uint8_t* output, int C, int H, int W);
void cuda_postprocess(const float* gpu_input, int output_c, int output_h, int output_w,
    uint8_t* cpu_mask_out);


