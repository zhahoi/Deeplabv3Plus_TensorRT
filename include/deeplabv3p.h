#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

class DeepLabV3P
{

public:

    DeepLabV3P(string model_path, nvinfer1::ILogger& logger);
    ~DeepLabV3P();

    void preprocess(Mat& image);
    void infer();
    void postprocess();
    void draw(const cv::Mat& image, const std::string& save_path);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;
    uint8_t* cpu_mask = nullptr;         // 分类结果（H X W)

    cudaStream_t stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int input_w;
    int input_h;
    int output_w;
    int output_h;
    int output_c;
    int output_size;
    const int MAX_IMAGE_SIZE = 2048 * 2048;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};