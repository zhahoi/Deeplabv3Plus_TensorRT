#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "deeplabv3p.h"

#include <filesystem> // c++17
namespace fs = std::filesystem;


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

bool isImageFile(const fs::path& path) {
    static const std::vector<std::string> imageExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    };
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(imageExtensions.begin(), imageExtensions.end(), ext) != imageExtensions.end();
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <input_path>" << std::endl;
        return -1;
    }

    const string engine_file_path{ argv[1] };
    const string input_path{ argv[2] };
    std::vector<fs::path> imagePathList;
    bool isVideo{ false };

    // 处理输入路径
    if (IsFile(input_path)) {
        fs::path file_path(input_path);
        if (isImageFile(file_path)) {
            imagePathList.push_back(file_path);
        }
        else if (file_path.extension() == ".mp4" ||
            file_path.extension() == ".avi" ||
            file_path.extension() == ".mov") {
            isVideo = true;
        }
        else {
            std::cerr << "Unsupported file format: " << file_path.extension() << std::endl;
            return -1;
        }
    }
    else if (IsPathExist(input_path)) {
        // 遍历目录获取所有图片
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                imagePathList.push_back(entry.path());
            }
        }
    }
    else {
        std::cerr << "Path does not exist: " << input_path << std::endl;
        return -1;
    }

    // Assume it's a folder, add logic to handle folders
     // 初始化模型
    DeepLabV3P model(engine_file_path, logger);

    // 创建输出目录
    const std::string output_dir = "output";
    if (!IsPathExist(output_dir)) {
        fs::create_directory(output_dir);
    }

    // 处理视频输入
    if (isVideo) {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file: " << input_path << std::endl;
            return -1;
        }

        // 创建视频写入器
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);

        fs::path video_path(input_path);
        std::string output_video = (fs::path(output_dir) /
            (video_path.stem().string() + "_result.mp4")).string();

        cv::VideoWriter writer(output_video,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(frame_width, frame_height));

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            // 处理帧
            model.preprocess(frame);
            auto start = std::chrono::high_resolution_clock::now();
            model.infer();
            auto end = std::chrono::high_resolution_clock::now();
            model.postprocess();

            // 计算处理时间
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Processing time: " << duration.count() << "ms" << std::endl;

            // 绘制结果并保存
            cv::Mat result_frame = frame.clone();
            model.draw(result_frame, "");
            writer.write(result_frame);

            cv::imshow("Result", result_frame);
            if (cv::waitKey(1) == 27) break;  // ESC键退出
        }

        cap.release();
        writer.release();
        cv::destroyAllWindows();
    }
    // 处理图片输入
    else {
        for (const auto& imagePath : imagePathList) {
            cv::Mat image = cv::imread(imagePath.string());
            if (image.empty()) {
                std::cerr << "Error reading image: " << imagePath << std::endl;
                continue;
            }

            // 处理图像
            model.preprocess(image);
            auto start = std::chrono::high_resolution_clock::now();
            model.infer();
            auto end = std::chrono::high_resolution_clock::now();
            model.postprocess();

            // 计算处理时间
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << imagePath.filename() << " processed in "
                << duration.count() << "ms" << std::endl;

            // 保存结果
            std::string save_name = (fs::path(output_dir) /
                (imagePath.stem().string() + "_result" + imagePath.extension().string())).string();

            model.draw(image, save_name);
            std::cout << "Result saved to: " << save_name << std::endl;
        }
    }

    return 0;
}
