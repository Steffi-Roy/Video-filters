/*
  Face embedding similarity demo.

  Usage:
    ./extensions <yunet.onnx> <sface.onnx> <image1> <image2>

  Example model files from OpenCV Zoo:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx

  Build:
    g++ extensions.cpp -I. -o extensions `pkg-config --cflags --libs opencv4`
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/face.hpp>

static int detectBestFace(cv::Ptr<cv::FaceDetectorYN> detector,
                          cv::Mat &image,
                          cv::Mat &bestFace) {
    detector->setInputSize(image.size());

    cv::Mat faces;
    int status = detector->detect(image, faces);
    if (status != 1 || faces.empty()) {
        return -1;
    }

    int bestIndex = 0;
    float bestScore = -1.0f;
    for (int i = 0; i < faces.rows; i++) {
        float score = faces.at<float>(i, 14);
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    bestFace = faces.row(bestIndex).clone();
    return 0;
}

static const char *interpretCosine(double score) {
    // OpenCV's SFace docs commonly use about 0.363 as a same-person
    // cosine threshold. Treat this as a demo threshold, not a guarantee.
    if (score >= 0.50) {
        return "very similar, likely same person";
    }
    if (score >= 0.363) {
        return "somewhat similar, possible same person";
    }
    return "not very similar";
}




int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <yunet.onnx> <sface.onnx> <image1> <image2>\n", argv[0]);
        printf("\n");
        printf("Suggested OpenCV Zoo models:\n");
        printf("  face_detection_yunet_2023mar.onnx\n");
        printf("  face_recognition_sface_2021dec.onnx\n");
        return -1;
    }

    const char *detectorModel = argv[1];
    const char *recognizerModel = argv[2];
    const char *imagePath1 = argv[3];
    const char *imagePath2 = argv[4];

    if (std::string(detectorModel) == std::string(recognizerModel)) {
        printf("The detector and recognizer model paths are identical.\n");
        printf("Use YuNet for argument 1 and SFace for argument 2.\n");
        return -1;
    }

    if (!std::filesystem::exists(detectorModel)) {
        printf("Could not find YuNet detector model: %s\n", detectorModel);
        printf("Pass the full path to face_detection_yunet_2023mar.onnx or put it in this folder.\n");
        return -1;
    }
    if (!std::filesystem::exists(recognizerModel)) {
        printf("Could not find SFace recognizer model: %s\n", recognizerModel);
        printf("Pass the full path to face_recognition_sface_2021dec.onnx or put it in this folder.\n");
        return -1;
    }

    cv::Mat image1 = cv::imread(imagePath1);
    cv::Mat image2 = cv::imread(imagePath2);

    if (image1.empty()) {
        printf("Could not read image: %s\n", imagePath1);
        return -1;
    }
    if (image2.empty()) {
        printf("Could not read image: %s\n", imagePath2);
        return -1;
    }

    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
        detectorModel,
        "",
        cv::Size(320, 320),
        0.85f,
        0.3f,
        5000
    );

    cv::Ptr<cv::FaceRecognizerSF> recognizer = cv::FaceRecognizerSF::create(
        recognizerModel,
        ""
    );

    cv::Mat face1;
    cv::Mat face2;
    if (detectBestFace(detector, image1, face1) != 0) {
        printf("No face detected in %s\n", imagePath1);
        return -1;
    }
    if (detectBestFace(detector, image2, face2) != 0) {
        printf("No face detected in %s\n", imagePath2);
        return -1;
    }

    cv::Mat aligned1;
    cv::Mat aligned2;
    recognizer->alignCrop(image1, face1, aligned1);
    recognizer->alignCrop(image2, face2, aligned2);

    cv::Mat feature1;
    cv::Mat feature2;
    recognizer->feature(aligned1, feature1);
    feature1 = feature1.clone();
    recognizer->feature(aligned2, feature2);
    feature2 = feature2.clone();

    double cosineScore = recognizer->match(
        feature1,
        feature2,
        cv::FaceRecognizerSF::FR_COSINE
    );

    double l2Score = recognizer->match(
        feature1,
        feature2,
        cv::FaceRecognizerSF::FR_NORM_L2
    );

    double diffNorm = cv::norm(feature1, feature2, cv::NORM_L2);

    printf("Image 1: %s\n", imagePath1);
    printf("Image 2: %s\n", imagePath2);
   
    printf("OpenCV cosine similarity: %f\n", cosineScore);
    
    printf("OpenCV L2 distance: %f\n", l2Score);
 
    printf("cv::norm feature difference: %f\n", diffNorm);
    printf("Interpretation: %s\n", interpretCosine(cosineScore));

    cv::imwrite("aligned_face_1.png", aligned1);
    cv::imwrite("aligned_face_2.png", aligned2);
    printf("Saved aligned_face_1.png and aligned_face_2.png\n");

    return 0;
}
