/*
Display Video and key mappings to show the filters
*/

#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "filter.h"
#include "faceDetect/faceDetect.h"
#include "DA2Network.hpp"

enum DisplayMode {
    ORIGINAL,
    OPENCV_GRAY,
    CUSTOM_GRAY,
    SEPIA,
    BLUR,
    SOBEL_X,
    SOBEL_Y,
    GRADIENT_MAGNITUDE,
    BLUR_QUANTIZE,
    NEGATIVE,
    EMBOSS,
    CARTOON,
    FACE_COLOR_POP,
    DEPTH_MAP,
    DEPTH_FOG
};

static void printHelp() {
    printf("\nControls:\n");
    printf("  q: quit\n");
    printf("  s: save current displayed frame\n");
    printf("  o: original color video\n");
    printf("  g: OpenCV grayscale\n");
    printf("  h: custom alternative grayscale\n");
    printf("  a: sepia with vignette\n");
    printf("  b: 5x5 blur\n");
    printf("  x: Sobel X absolute value\n");
    printf("  y: Sobel Y absolute value\n");
    printf("  m: gradient magnitude\n");
    printf("  l: blur and quantize\n");
    printf("  f: toggle face boxes\n");
    printf("  n: negative colors\n");
    printf("  e: emboss\n");
    printf("  t: cartoonization\n");
    printf("  c: face color pop\n");
    printf("  d: Depth Anything V2 depth map\n");
    printf("  p: depth-based fog\n");
    printf("  ?: show controls\n\n");
}

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

static void saveOutput(const char *filename, cv::Mat &image) {
    if (cv::imwrite(filename, image)) {
        printf("Saved %s\n", filename);
    } else {
        printf("Could not save %s\n", filename);
    }
}

static int saveAllFilters(cv::Mat &src) {
    if (src.empty()) {
        printf("Input image is empty\n");
        return -1;
    }

    cv::Mat output;
    cv::Mat gray;
    cv::Mat sx;
    cv::Mat sy;
    cv::Mat depth;
    std::vector<cv::Rect> faces;

    src.copyTo(output);
    saveOutput("original_filter.png", output);

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
    saveOutput("opencv_grayscale_filter.png", output);

    grayscale(src, output);
    saveOutput("grayscale_filter.png", output);

    sepia(src, output);
    saveOutput("sepia_filter.png", output);

    blur5x5_2(src, output);
    saveOutput("blur_filter.png", output);

    sobelX3x3(src, sx);
    cv::convertScaleAbs(sx, output);
    saveOutput("sobel_x_filter.png", output);

    sobelY3x3(src, sy);
    cv::convertScaleAbs(sy, output);
    saveOutput("sobel_y_filter.png", output);

    magnitude(sx, sy, output);
    saveOutput("gradient_magnitude_filter.png", output);

    blurQuantize(src, output, 10);
    saveOutput("blur_quantize_filter.png", output);

    negative(src, output);
    saveOutput("negative_filter.png", output);

    emboss(src, output);
    saveOutput("emboss_filter.png", output);

    cartoon(src, output, 12, 8);
    saveOutput("cartoon_filter.png", output);

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    detectFaces(gray, faces);

    src.copyTo(output);
    drawBoxes(output, faces);
    saveOutput("face_detection_filter.png", output);

    faceColorPop(src, output, faces);
    saveOutput("face_color_pop_filter.png", output);

    std::string depthModelPath = findDepthModelPath();
    if (std::filesystem::exists(depthModelPath)) {
        try {
            DA2Network daNet(depthModelPath.c_str());
            daNet.set_input(src, 0.4f);
            daNet.run_network(depth, src.size());
            cv::applyColorMap(depth, output, cv::COLORMAP_INFERNO);
            saveOutput("depth_map_filter.png", output);

            depthFog(src, depth, output);
            saveOutput("depth_fog_filter.png", output);
        } catch (const std::exception &e) {
            printf("Skipping depth filters: %s\n", e.what());
        }
    } else {
        printf("Skipping depth filters: model not found at %s\n", depthModelPath.c_str());
    }

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        cv::Mat src = cv::imread(argv[1]);
        if (src.empty()) {
            printf("Could not read image: %s\n", argv[1]);
            return -1;
        }
        printf("Saving all filters for %s\n", argv[1]);
        return saveAllFilters(src);
    }

    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    printHelp();

    cv::namedWindow("Video", 1);

    DisplayMode mode = ORIGINAL;
    bool showFaces = false;
    cv::Mat frame;
    cv::Mat output;
    cv::Mat gray;
    cv::Mat sx;
    cv::Mat sy;
    cv::Mat depth;
    cv::Mat depthColor;
    std::vector<cv::Rect> faces;
    std::unique_ptr<DA2Network> daNet;
    std::string depthModelPath = findDepthModelPath();
    const float depthScale = 0.4f; // resizes the image beofre sending to the model 
    int saveCount = 0;

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        char key = (char)cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == '?') {
            printHelp();
        } else if (key == 'o') {
            mode = ORIGINAL;
        } else if (key == 'g') {
            mode = OPENCV_GRAY;
        } else if (key == 'h') {
            mode = CUSTOM_GRAY;
        } else if (key == 'a') {
            mode = SEPIA;
        } else if (key == 'b') {
            mode = BLUR;
        } else if (key == 'x') {
            mode = SOBEL_X;
        } else if (key == 'y') {
            mode = SOBEL_Y;
        } else if (key == 'm') {
            mode = GRADIENT_MAGNITUDE;
        } else if (key == 'l') {
            mode = BLUR_QUANTIZE;
        } else if (key == 'n') {
            mode = NEGATIVE;
        } else if (key == 'e') {
            mode = EMBOSS;
        } else if (key == 't') {
            mode = CARTOON;
        } else if (key == 'c') {
            mode = FACE_COLOR_POP;
        } else if (key == 'd') {
            mode = DEPTH_MAP;
        } else if (key == 'p') {
            mode = DEPTH_FOG;
        } else if (key == 'f') {
            showFaces = !showFaces;
            printf("Face boxes %s\n", showFaces ? "on" : "off");
        }

        faces.clear();
        if (showFaces || mode == FACE_COLOR_POP) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            detectFaces(gray, faces);
        }

        switch (mode) {
            case OPENCV_GRAY:
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
                break;
            case CUSTOM_GRAY:
                grayscale(frame, output);
                break;
            case SEPIA:
                sepia(frame, output);
                break;
            case BLUR:
                blur5x5_2(frame, output);
                break;
            case SOBEL_X:
                sobelX3x3(frame, sx);
                cv::convertScaleAbs(sx, output);
                break;
            case SOBEL_Y:
                sobelY3x3(frame, sy);
                cv::convertScaleAbs(sy, output);
                break;
            case GRADIENT_MAGNITUDE:
                sobelX3x3(frame, sx);
                sobelY3x3(frame, sy);
                magnitude(sx, sy, output);
                break;
            case BLUR_QUANTIZE:
                blurQuantize(frame, output, 10);
                break;
            case NEGATIVE:
                negative(frame, output);
                break;
            case EMBOSS:
                emboss(frame, output);
                break;
            case CARTOON:
                cartoon(frame, output, 12, 8);
                break;
            case FACE_COLOR_POP:
                faceColorPop(frame, output, faces);
                break;
            case DEPTH_MAP:
                if (!daNet) {
                    daNet = std::make_unique<DA2Network>(depthModelPath.c_str());
                }

                daNet->set_input(frame, depthScale);
                daNet->run_network(depth, frame.size());
                cv::applyColorMap(depth, output, cv::COLORMAP_INFERNO);
                break;
            case DEPTH_FOG:
                if (!daNet) {
                    daNet = std::make_unique<DA2Network>(depthModelPath.c_str());
                }
                daNet->set_input(frame, depthScale);
                daNet->run_network(depth, frame.size());
                depthFog(frame, depth, output);
                break;
            case ORIGINAL:
            default:
                frame.copyTo(output);
                break;
        }

        if (showFaces) {
            drawBoxes(output, faces);
        }

        if (key == 's') {
            char filename[256];
            snprintf(filename, sizeof(filename), "saved_img_%03d.png", saveCount++);
            cv::imwrite(filename, output);
            printf("Image saved to %s\n", filename);
        }

        cv::imshow("Video", output);
    }

    return 0;
}
