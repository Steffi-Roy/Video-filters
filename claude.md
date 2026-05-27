# Git Notes

Use this file as a checklist for what should and should not be committed.

## Add These Files

These are the main source files and project files:

```bash
git add README.md
git add CMakeLists.txt
git add vidDisplay.cpp
git add filter.cpp
git add filter.h
git add blurTiming.cpp
git add depthImage.cpp
git add extensions.cpp
git add img_display.cpp
git add timeBlur.cpp
git add FILTER_CONCEPTS.md
git add report.tex
```

Add the face detection source and Haar cascade:

```bash
git add faceDetect/faceDetect.cpp
git add faceDetect/faceDetect.h
git add faceDetect/haarcascade_frontalface_alt2.xml
```

Add the Depth Anything wrapper:

```bash
git add da2-code/DA2Network.hpp
```

Add small input/example images only if they are needed for the report or grading:

```bash
git add pathfinder.png
git add extracted_pages_images/*.png
```

Add the small YuNet detector model if required:

```bash
git add face_detection_yunet_2023mar.onnx
```

## Be Careful With Large Model Files

These files are large. Only commit them if the course requires model files to be submitted in the repository:

```text
da2-code/model_fp16.onnx
da2-code.zip
face_recognition_sface_2021dec.onnx
```

If using GitHub, it is usually better to avoid committing these large files and instead document where to download them.

## Ignore These Files

Do not commit build outputs, compiled binaries, or generated screenshots:

```gitignore
build/
.DS_Store
.vscode/

app
vid
vidTest
timeBlurTest
extensions
extensi
extension

saved_img*.png
*_filter.png
aligned_face_*.png

*.o
```

## Suggested Commit Flow

```bash
git status
git add README.md CMakeLists.txt vidDisplay.cpp filter.cpp filter.h
git add blurTiming.cpp depthImage.cpp extensions.cpp img_display.cpp timeBlur.cpp
git add faceDetect/faceDetect.cpp faceDetect/faceDetect.h faceDetect/haarcascade_frontalface_alt2.xml
git add da2-code/DA2Network.hpp
git add FILTER_CONCEPTS.md report.tex claude.md
git status
git commit -m "Implement real-time computer vision filters"
git push
```

