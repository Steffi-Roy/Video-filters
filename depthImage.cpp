#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "DA2Network.hpp"
#include "filter.h"

static std::string findDepthModelPath() {
    const char *paths[] = {
        "da2-code/model_fp16.onnx",
        "../da2-code/model_fp16.onnx",
        "model_fp16.onnx"
    };

    for (const char *path : paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return "da2-code/model_fp16.onnx";
}

int main(int argc, char **argv) {
    const char *filename = argc > 1 ? argv[1] : "pathfinder.png";
    cv::Mat src = cv::imread(filename);
    if (src.empty()) {
        printf("Unable to read image: %s\n", filename);
        return -1;
    }

    std::string modelPath = findDepthModelPath();
    DA2Network daNet(modelPath.c_str());

    int minSide = std::min(src.rows, src.cols);
    float scale = std::min(1.0f, 384.0f / minSide);
    cv::Mat depth;
    cv::Mat depthColor;
    cv::Mat fog;

    daNet.set_input(src, scale);
    daNet.run_network(depth, src.size());
    cv::applyColorMap(depth, depthColor, cv::COLORMAP_INFERNO);
    depthFog(src, depth, fog);

    cv::imwrite("depth_image.png", depthColor);
    cv::imwrite("depth_fog.png", fog);
    printf("Wrote depth_image.png and depth_fog.png\n");

    return 0;
}
