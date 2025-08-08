#include "deeplabv3p.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "postprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>

static Logger logger;
#define isFP16 true
#define warmup true

DeepLabV3P::DeepLabV3P(string model_path, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


DeepLabV3P::~DeepLabV3P()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;
    delete[] cpu_mask;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}


void DeepLabV3P::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // Read the engine file（读取.engine文件），把TensorRT序列化模型从磁盘加载到内存中
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine（反序列化模型，创建执行上下文）
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model （获取模型输入维度信息）
    input_h = engine->getBindingDimensions(0).d[2];  // 512
    input_w = engine->getBindingDimensions(0).d[3];  // 512

    // 获取模型输出维度信息
    output_c = engine->getBindingDimensions(1).d[1]; // 检测结果的属性数量 21
    output_h = engine->getBindingDimensions(1).d[2];  // 512
    output_w = engine->getBindingDimensions(1).d[3];  // 512
 
    // Initialize input buffers (分配输入和输出缓冲区 CPU + GPU)
    cpu_output_buffer = new float[output_c * output_h * output_w];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], output_c * output_h * output_w * sizeof(float)));

    cpu_mask = new uint8_t[output_h * output_w];

    // 初始化CUDA预处理资源
    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 模型热身（warmup)
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}


void DeepLabV3P::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
}


void DeepLabV3P::preprocess(Mat& image) {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


void DeepLabV3P::postprocess()
{
    float* gpu_output = static_cast<float*>(gpu_buffers[1]);
    cuda_postprocess(gpu_output, output_c, output_h, output_w, cpu_mask);
}


void DeepLabV3P::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}


bool DeepLabV3P::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}


void DeepLabV3P::draw(const cv::Mat& image, const std::string& save_path)
{
    cv::Mat color_mask(output_h, output_w, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
            int class_id = cpu_mask[y * output_w + x];
            cv::Vec3b color;

            if (class_id >= 0 && class_id < COLORS.size()) {
                auto& c = COLORS[class_id];
                color = cv::Vec3b(c[0], c[1], c[2]);  // 注意BGR顺序
            }
            else {
                color = cv::Vec3b(0, 0, 0);
            }
            color_mask.at<cv::Vec3b>(y, x) = color;
        }
    }

    // Resize到原图大小
    cv::Mat mask_resized;
    cv::resize(color_mask, mask_resized, image.size());

    // 混合原图和mask
    cv::Mat blended;
    cv::addWeighted(image, 0.5, mask_resized, 0.5, 0, blended);

    // 显示和保存
    // imshow("Prediction", blended);
    // waitKey(0);

    if (!save_path.empty()) {
        imwrite(save_path, blended);
    }
}