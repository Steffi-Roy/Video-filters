#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "filter.h"

static double timeBlur(int (*blurFn)(cv::Mat &, cv::Mat &), cv::Mat &src, int iterations) {
    cv::Mat dst;
    double start = static_cast<double>(cv::getTickCount());
    for (int i = 0; i < iterations; i++) {
        blurFn(src, dst);
    }
    double end = static_cast<double>(cv::getTickCount());
    return 1000.0 * (end - start) / cv::getTickFrequency() / iterations;
}

int main(int argc, char **argv) {
    const char *filename = argc > 1 ? argv[1] : "pathfinder.png";
    cv::Mat src = cv::imread(filename);
    if (src.empty()) {
        printf("Unable to read image: %s\n", filename);
        return -1;
    }

    int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 50;
    double blur1Ms = timeBlur(blur5x5_1, src, iterations);
    double blur2Ms = timeBlur(blur5x5_2, src, iterations);

    printf("Image: %s (%d x %d)\n", filename, src.cols, src.rows);
    printf("Iterations: %d\n", iterations);
    printf("blur5x5_1 average: %.3f ms\n", blur1Ms);
    printf("blur5x5_2 average: %.3f ms\n", blur2Ms);
    printf("Speedup: %.2fx\n", blur1Ms / blur2Ms);

    return 0;
}
