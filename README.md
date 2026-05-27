# Real-Time Computer Vision Filters

This project implements a real-time OpenCV video filtering application in C++. It supports live webcam filtering, still-image batch filtering, face detection, depth-based effects with Depth Anything V2, cartoonization, and a face similarity extension using face embeddings.

## Build

This project uses CMake.

```bash
cd "/Users/steffi/Documents/NU Courses/CV"
cmake -S . -B build
cmake --build build
```

You can also build a specific target:

```bash
cmake --build build --target vid
cmake --build build --target blurTiming
cmake --build build --target depthImage
cmake --build build --target extensions
```

## Main Video Program

Run the live webcam application:

```bash
./build/vid
```

### Keyboard Controls

| Key | Effect |
|---|---|
| `o` | Original video |
| `g` | OpenCV grayscale |
| `h` | Custom grayscale |
| `a` | Sepia with vignette |
| `b` | 5x5 blur |
| `x` | Sobel X |
| `y` | Sobel Y |
| `m` | Gradient magnitude |
| `l` | Blur and quantize |
| `f` | Toggle face detection boxes |
| `n` | Negative colors |
| `e` | Emboss |
| `t` | Cartoonization |
| `c` | Face color pop |
| `d` | Depth Anything V2 depth map |
| `p` | Depth-based fog |
| `s` | Save current displayed frame |
| `q` | Quit |

Saved frames are written as:

```text
saved_img_000.png
saved_img_001.png
saved_img_002.png
...
```

## Batch Image Mode

You can also apply all filters to a still image at once:

```bash
./build/vid pathfinder.png
```

or:

```bash
./build/vid /path/to/your/image.jpg
```

This writes named output files:

```text
original_filter.png
opencv_grayscale_filter.png
grayscale_filter.png
sepia_filter.png
blur_filter.png
sobel_x_filter.png
sobel_y_filter.png
gradient_magnitude_filter.png
blur_quantize_filter.png
negative_filter.png
emboss_filter.png
cartoon_filter.png
face_detection_filter.png
face_color_pop_filter.png
depth_map_filter.png
depth_fog_filter.png
```

## Blur Timing

The blur timing program compares the naive 5x5 blur with the faster separable implementation.

```bash
./build/blurTiming pathfinder.png 50
```

Example result:

```text
blur5x5_1 average: 694.959 ms
blur5x5_2 average: 113.624 ms
Speedup: 6.12x
```

## Depth Anything V2

Depth Anything V2 is used through ONNX Runtime. It estimates a relative depth value for each pixel from a normal RGB image.

Required file:

```text
da2-code/model_fp16.onnx
```

The project uses:

```text
da2-code/DA2Network.hpp
```

Depth modes in `vid`:

| Key | Effect |
|---|---|
| `d` | Colorized depth map |
| `p` | Depth-based fog |

You can also generate still-image depth outputs:

```bash
./build/depthImage pathfinder.png
```

This writes:

```text
depth_image.png
depth_fog.png
```

## Face Similarity Extension

The face similarity extension uses OpenCV's YuNet face detector and SFace face recognizer.

Build:

```bash
cmake --build build --target extensions
```

Run:

```bash
./build/extensions \
  face_detection_yunet_2023mar.onnx \
  face_recognition_sface_2021dec.onnx \
  image1.jpg \
  image2.jpg
```

The program:

1. Detects the strongest face in each image.
2. Aligns and crops each face.
3. Extracts a face embedding vector from each aligned face.
4. Compares the embeddings using cosine similarity and L2 distance.

Higher cosine similarity means the faces are more similar. Lower L2 distance means the embeddings are closer in feature space.

## Important Files

| File | Purpose |
|---|---|
| `vidDisplay.cpp` | Main video application and batch image mode |
| `filter.cpp` | Custom image filters |
| `filter.h` | Filter function prototypes |
| `blurTiming.cpp` | Timing comparison for blur implementations |
| `depthImage.cpp` | Still-image depth output generator |
| `extensions.cpp` | Face similarity extension |
| `faceDetect/` | Haar cascade face detection support files |
| `da2-code/DA2Network.hpp` | Depth Anything V2 wrapper |
| `FILTER_CONCEPTS.md` | Explanation of filter concepts and code |
| `report.tex` | Project report source |

## Notes For Git

Do not commit build outputs or generated images unless they are required for submission.

Recommended ignored files:

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

Large model files such as `da2-code/model_fp16.onnx`, `da2-code.zip`, and `face_recognition_sface_2021dec.onnx` may be too large for a normal GitHub repository. If they are not committed, include download instructions or submit them separately as required by the course.

