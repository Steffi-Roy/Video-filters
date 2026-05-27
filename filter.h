#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>

int greyscale(cv::Mat &src, cv::Mat &dst);
int grayscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int negative(cv::Mat &src, cv::Mat &dst);
int emboss(cv::Mat &src, cv::Mat &dst);
int cartoon(cv::Mat &src, cv::Mat &dst, int levels = 15, int edgeThreshold = 55);
int faceColorPop(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst);

#endif
