/*
Implemented filter functions such as grayscale, sepia, 5x5 blur, (X,Y)Sobel, gradient magnitude, blur, facedetect, Depth anything, fog, cartoon
*/

/* to compile and build this file run this cmd: g++ -std=c++17 vidDisplay.cpp filter.cpp faceDetect/faceDetect.cpp \
-I. -Ida2-code \
-o vidTest \
`pkg-config --cflags --libs opencv4 libonnxruntime`*/

/*To run in batch mode for an image run ./vidTest <path to image>*/


#include "filter.h"

#include <algorithm>
#include <cmath>

static unsigned char clampToUchar(int value) {
    return static_cast<unsigned char>(std::max(0, std::min(255, value)));
}

int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_8UC3);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int blue = srcRow[j][0];
            int green = srcRow[j][1];
            int red = srcRow[j][2];
            int value = clampToUchar(255 - red + (green - blue) / 3);
            dstRow[j] = cv::Vec3b(value, value, value);
        }
    }

    return 0;
}

int grayscale(cv::Mat &src, cv::Mat &dst) {
    return greyscale(src, dst);
}

int sepia(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_8UC3);
    double cx = (src.cols - 1) / 2.0;
    double cy = (src.rows - 1) / 2.0;
    double maxDist = std::sqrt(cx * cx + cy * cy);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int b = srcRow[j][0];
            int g = srcRow[j][1];
            int r = srcRow[j][2];

            int newB = static_cast<int>(0.272 * r + 0.534 * g + 0.131 * b);
            int newG = static_cast<int>(0.349 * r + 0.686 * g + 0.168 * b);
            int newR = static_cast<int>(0.393 * r + 0.769 * g + 0.189 * b);

            double dx = j - cx;
            double dy = i - cy;
            double vignette = 1.0 - 0.75 * std::pow((std::sqrt(dx * dx + dy * dy) / maxDist), 2.0);
            //need to clamp as the newR and newG coefficient go above 1, or else give blue tint.
            dstRow[j][0] = clampToUchar(static_cast<int>(newB * vignette));
            dstRow[j][1] = clampToUchar(static_cast<int>(newG * vignette));
            dstRow[j][2] = clampToUchar(static_cast<int>(newR * vignette));
        }
    }

    return 0;
}

int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    src.copyTo(dst);

    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            int sum[3] = {0, 0, 0};
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    int weight = kernel[ki + 2][kj + 2];
                    for (int c = 0; c < 3; c++) {
                        sum[c] += pixel[c] * weight;
                    }
                }
            }
            cv::Vec3b &out = dst.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++) {
                out[c] = clampToUchar(sum[c] / 100);
            }
        }
    }
    

    return 0;
}

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat temp(src.size(), CV_16SC3);
    src.copyTo(dst);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tmpRow = temp.ptr<cv::Vec3s>(i);
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                tmpRow[j][c] = srcRow[j - 2][c] + 2 * srcRow[j - 1][c] +
                               4 * srcRow[j][c] + 2 * srcRow[j + 1][c] +
                               srcRow[j + 2][c];
            }
        }
    }

    for (int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        const cv::Vec3s *rowM2 = temp.ptr<cv::Vec3s>(i - 2);
        const cv::Vec3s *rowM1 = temp.ptr<cv::Vec3s>(i - 1);
        const cv::Vec3s *row = temp.ptr<cv::Vec3s>(i);
        const cv::Vec3s *rowP1 = temp.ptr<cv::Vec3s>(i + 1);
        const cv::Vec3s *rowP2 = temp.ptr<cv::Vec3s>(i + 2);
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                int value = rowM2[j][c] + 2 * rowM1[j][c] + 4 * row[j][c] +
                            2 * rowP1[j][c] + rowP2[j][c];
                dstRow[j][c] = clampToUchar(value / 100);
            }
        }
    }

    return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tmpRow = temp.ptr<cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tmpRow[j][c] = srcRow[j + 1][c] - srcRow[j - 1][c];
            }
        }
    }

    for (int i = 1; i < src.rows - 1; i++) {
        const cv::Vec3s *rowM1 = temp.ptr<cv::Vec3s>(i - 1);
        const cv::Vec3s *row = temp.ptr<cv::Vec3s>(i);
        const cv::Vec3s *rowP1 = temp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = rowM1[j][c] + 2 * row[j][c] + rowP1[j][c];
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows - 1; i++) {
        const cv::Vec3b *rowM1 = src.ptr<cv::Vec3b>(i - 1);
        const cv::Vec3b *rowP1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3s *tmpRow = temp.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                tmpRow[j][c] = rowM1[j][c] - rowP1[j][c];
            }
        }
    }

    for (int i = 1; i < src.rows - 1; i++) {
        const cv::Vec3s *tmpRow = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = tmpRow[j - 1][c] + 2 * tmpRow[j][c] + tmpRow[j + 1][c];
            }
        }
    }

    return 0;
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty() || sx.size() != sy.size()) {
        return -1;
    }

    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        const cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        const cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int x = sxRow[j][c];
                int y = syRow[j][c];
                dstRow[j][c] = clampToUchar(static_cast<int>(std::sqrt(x * x + y * y)));
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || levels < 2) {
        return -1;
    }

    cv::Mat blurred;
    blur5x5_2(src, blurred);
    dst.create(src.size(), CV_8UC3);

    int bucket = std::max(1, 255 / levels);
    for (int i = 0; i < blurred.rows; i++) {
        const cv::Vec3b *srcRow = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < blurred.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int xt = srcRow[j][c] / bucket;
                dstRow[j][c] = clampToUchar(xt * bucket);
            }
        }
    }

    return 0;
}

int negative(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            dstRow[j] = cv::Vec3b(255 - srcRow[j][0], 255 - srcRow[j][1], 255 - srcRow[j][2]);
        }
    }

    return 0;
}

int emboss(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat sx;
    cv::Mat sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    dst.create(src.size(), CV_8UC3);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        const cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value = 128 + static_cast<int>(0.35 * sxRow[j][c] + 0.35 * syRow[j][c]);
                dstRow[j][c] = clampToUchar(value);
            }
        }
    }

    return 0;
}

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int edgeThreshold) {
    if (src.empty() || levels < 2) {
        return -1;
    }

    cv::Mat smooth;
    src.copyTo(smooth);

    for (int i = 0; i < 4; i++) {
        cv::Mat next;
        cv::bilateralFilter(smooth, next, 7, 45, 5);
        smooth = next;
    }

    cv::Mat gray;
    cv::Mat smallBlur;
    cv::Mat largeBlur;
    cv::Mat dog;
    cv::cvtColor(smooth, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, smallBlur, cv::Size(0, 0), 1.0);
    cv::GaussianBlur(gray, largeBlur, cv::Size(0, 0), 1.6);
    cv::subtract(smallBlur, largeBlur, dog, cv::noArray(), CV_16S);

    dst.create(src.size(), CV_8UC3);
    int bucket = std::max(1, 255 / levels);

    for (int i = 0; i < smooth.rows; i++) {
        const cv::Vec3b *smoothRow = smooth.ptr<cv::Vec3b>(i);
        const short *dogRow = dog.ptr<short>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < smooth.cols; j++) {
            int edgeStrength = std::abs(dogRow[j]);

            if (edgeStrength > edgeThreshold) {
                dstRow[j] = cv::Vec3b(0, 0, 0);
                continue;
            }

            for (int c = 0; c < 3; c++) {
                int x = smoothRow[j][c];
                int nearest = (x / bucket) * bucket;
                int next = std::min(255, nearest + bucket);
                double t = (x - nearest) / static_cast<double>(bucket);
                double softStep = 0.5 + 0.5 * std::tanh(6.0 * (t - 0.5));
                dstRow[j][c] = clampToUchar(static_cast<int>(nearest * (1.0 - softStep) + next * softStep));
            }
        }
    }

    return 0;
}

int faceColorPop(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces) {
    if (src.empty()) {
        return -1;
    }

    greyscale(src, dst);

    for (const cv::Rect &faceRect : faces) {
        cv::Rect bounded = faceRect & cv::Rect(0, 0, src.cols, src.rows);
        if (bounded.width <= 0 || bounded.height <= 0) {
            continue;
        }
        src(bounded).copyTo(dst(bounded));
    }

    return 0;
}

int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst) {
    if (src.empty() || depth.empty() || src.size() != depth.size()) {
        return -1;
    }

    dst.create(src.size(), CV_8UC3);
    cv::Vec3b fogColor(225, 220, 205);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        const unsigned char *depthRow = depth.ptr<unsigned char>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            double d = depthRow[j] / 255.0;
            double fogAmount = 1.0 - std::exp(-2.2 * d * d);
            for (int c = 0; c < 3; c++) {
                double value = srcRow[j][c] * (1.0 - fogAmount) + fogColor[c] * fogAmount;
                dstRow[j][c] = clampToUchar(static_cast<int>(value));
            }
        }
    }

    return 0;
}
