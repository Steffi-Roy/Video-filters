# Filter Concepts and Code Guide

This guide explains the main ideas behind the filters implemented in this project. The big idea is that an image is just a grid of pixels. Each pixel has a position and a color.

For a color OpenCV image:

```cpp
int i; // row, y position
int j; // column, x position

cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
unsigned char b = pixel[0];
unsigned char g = pixel[1];
unsigned char r = pixel[2];
```

OpenCV stores color images as **BGR**, not RGB:

```text
pixel[0] = blue
pixel[1] = green
pixel[2] = red
```

Most filters answer this question:

```text
Given the old pixel, its position, and maybe nearby pixels,
what should the new pixel be?
```

---

## 1. Basic Pixel Loop

Most pixel-wise filters use this pattern:

```cpp
dst.create(src.size(), CV_8UC3);

for (int i = 0; i < src.rows; i++) {
    const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

    for (int j = 0; j < src.cols; j++) {
        int b = srcRow[j][0];
        int g = srcRow[j][1];
        int r = srcRow[j][2];

        // compute newB, newG, newR

        dstRow[j][0] = newB;
        dstRow[j][1] = newG;
        dstRow[j][2] = newR;
    }
}
```

The row-pointer version is faster than repeatedly calling `at`, because it avoids extra indexing overhead.

---

## 2. Clamping

Pixel values must stay in the range `0` to `255`. Many formulas can produce values outside that range, so we clamp them:

```cpp
static unsigned char clampToUchar(int value) {
    return static_cast<unsigned char>(std::max(0, std::min(255, value)));
}
```

Example:

```cpp
dstRow[j][2] = clampToUchar(newR);
```

Without clamping, a value like `300` or `-20` can wrap around and create strange colors.

---

## 3. Custom Greyscale

### Concept

Greyscale means all three color channels are equal:

```text
B = G = R
```

OpenCV greyscale uses a standard luminance formula. Our custom greyscale intentionally looks different:

```cpp
int value = clampToUchar(255 - red + (green - blue) / 4);
dstRow[j] = cv::Vec3b(value, value, value);
```

### Why It Works

The output is still greyscale because each channel receives the same value. But the value is not the standard brightness. It is based mostly on inverse red, with a small green-blue contrast adjustment.

### Key Code

```cpp
int blue = srcRow[j][0];
int green = srcRow[j][1];
int red = srcRow[j][2];

int value = clampToUchar(255 - red + (green - blue) / 4);
dstRow[j] = cv::Vec3b(value, value, value);
```

---

## 4. Sepia Tone

### Concept

Sepia makes an image look warmer and antique. Each output channel is a weighted combination of the original red, green, and blue.

The important rule is:

```text
Use the original B, G, R values to compute all new channels.
Do not compute newR and then use newR to compute newG.
```

### Formula

```cpp
int newB = 0.272 * r + 0.534 * g + 0.131 * b;
int newG = 0.349 * r + 0.686 * g + 0.168 * b;
int newR = 0.393 * r + 0.769 * g + 0.189 * b;
```

### Key Code

```cpp
int b = srcRow[j][0];
int g = srcRow[j][1];
int r = srcRow[j][2];

int newB = static_cast<int>(0.272 * r + 0.534 * g + 0.131 * b);
int newG = static_cast<int>(0.349 * r + 0.686 * g + 0.168 * b);
int newR = static_cast<int>(0.393 * r + 0.769 * g + 0.189 * b);

dstRow[j][0] = clampToUchar(newB);
dstRow[j][1] = clampToUchar(newG);
dstRow[j][2] = clampToUchar(newR);
```

---

## 5. Vignette

### Concept

A vignette darkens pixels farther from the center of the image.

Each pixel has:

```text
position = (j, i)
center = (cx, cy)
```

The distance from the pixel to the center is:

```cpp
sqrt(dx * dx + dy * dy)
```

where:

```cpp
dx = j - cx;
dy = i - cy;
```

### Why `cx * cx + cy * cy`?

This calculates the distance from the image center to a corner using the Pythagorean theorem:

```cpp
double maxDist = std::sqrt(cx * cx + cy * cy);
```

Then we normalize each pixel distance:

```cpp
double amount = distance / maxDist;
```

So:

```text
center: amount ≈ 0
corner: amount ≈ 1
```

### Key Code

```cpp
double cx = (src.cols - 1) / 2.0;
double cy = (src.rows - 1) / 2.0;
double maxDist = std::sqrt(cx * cx + cy * cy);

double dx = j - cx;
double dy = i - cy;
double dist = std::sqrt(dx * dx + dy * dy);
double amount = dist / maxDist;

double vignette = 1.0 - 0.45 * std::pow(amount, 2.0);
```

Then multiply the color by the vignette factor:

```cpp
dstRow[j][0] = clampToUchar(newB * vignette);
dstRow[j][1] = clampToUchar(newG * vignette);
dstRow[j][2] = clampToUchar(newR * vignette);
```

---

## 6. Negative

### Concept

Negative color inverts every color channel.

```text
new value = 255 - old value
```

This is a single-pixel filter because it only needs the current pixel.

### Key Code

```cpp
dstRow[j] = cv::Vec3b(
    255 - srcRow[j][0],
    255 - srcRow[j][1],
    255 - srcRow[j][2]
);
```

---

## 7. Blur

### Concept

Blur replaces each pixel with a weighted average of nearby pixels. Instead of looking at just one pixel, blur looks at a neighborhood.

For a 5x5 blur, the output pixel depends on a 5x5 window:

```text
current pixel plus 2 pixels in every direction
```

The Gaussian-like kernel is:

```text
1  2  4  2  1
2  4  8  4  2
4  8 16  8  4
2  4  8  4  2
1  2  4  2  1
```

The weights sum to `100`, so we divide by `100`.

### Naive Version

```cpp
for (int ki = -2; ki <= 2; ki++) {
    for (int kj = -2; kj <= 2; kj++) {
        cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
        int weight = kernel[ki + 2][kj + 2];

        for (int c = 0; c < 3; c++) {
            sum[c] += pixel[c] * weight;
        }
    }
}

dst.at<cv::Vec3b>(i, j)[c] = sum[c] / 100;
```

This works, but it is slower because it uses nested loops and `at`.

### Faster Separable Version

The 5x5 kernel can be separated into:

```text
[1 2 4 2 1] horizontally
then
[1 2 4 2 1] vertically
```

This reduces work from 25 weighted samples per pixel to about 10.

Horizontal pass:

```cpp
tmpRow[j][c] =
    srcRow[j - 2][c] +
    2 * srcRow[j - 1][c] +
    4 * srcRow[j][c] +
    2 * srcRow[j + 1][c] +
    srcRow[j + 2][c];
```

Vertical pass:

```cpp
int value =
    rowM2[j][c] +
    2 * rowM1[j][c] +
    4 * row[j][c] +
    2 * rowP1[j][c] +
    rowP2[j][c];

dstRow[j][c] = clampToUchar(value / 100);
```

### Practical Lesson

The faster blur is faster because:

```text
fewer operations + pointer access + separable math
```

---

## 8. Sobel X and Sobel Y

### Concept

Sobel filters detect edges by measuring how quickly pixel values change.

If brightness changes sharply, there is probably an edge.

Sobel X detects left-right changes, so it highlights vertical edges.

Sobel Y detects up-down changes, so it highlights horizontal edges.

### Why Signed Short?

Edges can be positive or negative:

```text
dark-to-light = positive
light-to-dark = negative
```

So the output uses:

```cpp
CV_16SC3
```

That means:

```text
16-bit signed short, 3 channels
```

### Sobel X

Sobel X uses:

```text
[-1 0 1] horizontally
[ 1 2 1] vertically
```

Horizontal derivative:

```cpp
tmpRow[j][c] = srcRow[j + 1][c] - srcRow[j - 1][c];
```

Vertical smoothing:

```cpp
dstRow[j][c] = rowM1[j][c] + 2 * row[j][c] + rowP1[j][c];
```

### Sobel Y

Sobel Y uses:

```text
[ 1  2  1] horizontally
[ 1  0 -1] vertically
```

Vertical derivative:

```cpp
tmpRow[j][c] = rowM1[j][c] - rowP1[j][c];
```

Horizontal smoothing:

```cpp
dstRow[j][c] = tmpRow[j - 1][c] + 2 * tmpRow[j][c] + tmpRow[j + 1][c];
```

---

## 9. Gradient Magnitude

### Concept

Sobel X tells us horizontal change.

Sobel Y tells us vertical change.

Gradient magnitude combines them:

```text
magnitude = sqrt(sx^2 + sy^2)
```

This gives the overall edge strength.

### Key Code

```cpp
int x = sxRow[j][c];
int y = syRow[j][c];

dstRow[j][c] = clampToUchar(
    static_cast<int>(std::sqrt(x * x + y * y))
);
```

---

## 10. Blur and Quantize

### Concept

This effect first blurs the image, then reduces the number of color values.

Quantization means many nearby color values collapse into one bucket.

Example with `levels = 10`:

```cpp
int bucket = 255 / levels;
```

Then:

```cpp
int xt = x / bucket;
int xf = xt * bucket;
```

So values like `128`, `130`, and `134` might all become `125`.

### Key Code

```cpp
cv::Mat blurred;
blur5x5_2(src, blurred);

int bucket = std::max(1, 255 / levels);

int xt = srcRow[j][c] / bucket;
dstRow[j][c] = clampToUchar(xt * bucket);
```

### Why Blur First?

Blurring reduces noise and tiny details before the color buckets are applied. This makes the final image look cleaner and more stylized.

---

## 11. Face Color Pop

### Concept

This effect uses face detection.

The whole image becomes greyscale, but detected face rectangles are copied back in color.

### Key Code

```cpp
greyscale(src, dst);

for (const cv::Rect &faceRect : faces) {
    cv::Rect bounded = faceRect & cv::Rect(0, 0, src.cols, src.rows);
    src(bounded).copyTo(dst(bounded));
}
```

### Practical Meaning

This is a mask-based filter:

```text
outside face: greyscale
inside face: original color
```

---

## 12. Emboss

### Concept

Emboss makes edges look raised or carved.

It uses Sobel X and Sobel Y, then combines them as if light were shining from a diagonal direction.

Flat regions become gray.

Edges become lighter or darker depending on their direction.

### Key Code

```cpp
sobelX3x3(src, sx);
sobelY3x3(src, sy);

int value = 128 + static_cast<int>(
    0.35 * sxRow[j][c] + 0.35 * syRow[j][c]
);

dstRow[j][c] = clampToUchar(value);
```

### Why Add 128?

Sobel values can be negative or positive.

Adding `128` shifts the result into visible gray image range:

```text
negative edge -> darker than gray
zero edge     -> gray
positive edge -> lighter than gray
```

---

## 13. Depth Map

### Concept

Depth Anything V2 estimates a relative depth value for every pixel from a normal RGB image.

It outputs a greyscale image:

```text
dark/bright values represent relative depth
```

This is not exact physical distance. It is an estimate based on visual patterns.

### Practical Use

Once you have a depth map, you can make filters depend on distance:

```text
near pixels: one effect
far pixels: another effect
```

---

## 14. Depth Fog

### Concept

Fog increases with distance.

The code takes the depth value:

```cpp
double d = depthRow[j] / 255.0;
```

Now `d` is normalized:

```text
0.0 to 1.0
```

Then it creates an exponential fog amount:

```cpp
double fogAmount = 1.0 - std::exp(-2.2 * d * d);
```

### Blend Formula

The final pixel is a mix of:

```text
original color
fog color
```

```cpp
double value =
    srcRow[j][c] * (1.0 - fogAmount) +
    fogColor[c] * fogAmount;
```

If `fogAmount` is small, the output is mostly original color.

If `fogAmount` is large, the output is mostly fog color.

---

## 15. Cartoonization

### Concept

The cartoon filter is based on three ideas from real-time video abstraction:

1. Smooth small details while preserving edges.
2. Add strong dark edges.
3. Reduce the number of colors.

### Step 1: Bilateral Smoothing

Bilateral filtering smooths flat regions but tries to preserve strong edges.

```cpp
for (int i = 0; i < 4; i++) {
    cv::Mat next;
    cv::bilateralFilter(smooth, next, 7, 45, 5);
    smooth = next;
}
```

### Step 2: Difference of Gaussians Edges

Blur the same image at two scales:

```cpp
cv::GaussianBlur(gray, smallBlur, cv::Size(0, 0), 1.0);
cv::GaussianBlur(gray, largeBlur, cv::Size(0, 0), 1.6);
```

Subtract them:

```cpp
cv::subtract(smallBlur, largeBlur, dog, cv::noArray(), CV_16S);
```

Large difference means likely edge.

### Step 3: Soft Quantization

Instead of harsh bucket jumps, use a smooth transition:

```cpp
int nearest = (x / bucket) * bucket;
int next = std::min(255, nearest + bucket);
double t = (x - nearest) / static_cast<double>(bucket);
double softStep = 0.5 + 0.5 * std::tanh(6.0 * (t - 0.5));

dstRow[j][c] = clampToUchar(
    static_cast<int>(nearest * (1.0 - softStep) + next * softStep)
);
```

### Edge Overlay

If the DoG edge strength is high, draw black:

```cpp
if (edgeStrength > edgeThreshold) {
    dstRow[j] = cv::Vec3b(0, 0, 0);
    continue;
}
```

---

## 16. How to Design Your Own Filter

Ask these questions:

### 1. Does the filter depend only on the current pixel?

Examples:

```text
negative
greyscale
brightness
sepia
```

Pattern:

```cpp
newPixel = f(oldPixel);
```

### 2. Does the filter depend on pixel position?

Examples:

```text
vignette
spotlight
radial color shift
circle mask
```

Pattern:

```cpp
newPixel = f(oldPixel, i, j);
```

### 3. Does the filter depend on nearby pixels?

Examples:

```text
blur
Sobel
emboss
edge detection
sharpen
```

Pattern:

```cpp
newPixel = f(neighborhood around i, j);
```

### 4. Does the filter depend on another signal?

Examples:

```text
face detector
depth map
motion mask
segmentation mask
```

Pattern:

```cpp
newPixel = f(oldPixel, maskPixel or detected region);
```

---

## 17. Tiny Experiments to Try

### Brightness

```cpp
dstRow[j][0] = clampToUchar(b + 40);
dstRow[j][1] = clampToUchar(g + 40);
dstRow[j][2] = clampToUchar(r + 40);
```

### Contrast

```cpp
double contrast = 1.3;
dstRow[j][0] = clampToUchar((b - 128) * contrast + 128);
dstRow[j][1] = clampToUchar((g - 128) * contrast + 128);
dstRow[j][2] = clampToUchar((r - 128) * contrast + 128);
```

### Keep Only Red Areas in Color

```cpp
if (r > g * 1.4 && r > b * 1.4) {
    dstRow[j] = srcRow[j];
} else {
    int gray = (r + g + b) / 3;
    dstRow[j] = cv::Vec3b(gray, gray, gray);
}
```

### Circular Spotlight

```cpp
double dx = j - cx;
double dy = i - cy;
double dist = std::sqrt(dx * dx + dy * dy);
double amount = dist / maxDist;
double factor = 1.2 - 0.7 * amount;

dstRow[j][0] = clampToUchar(b * factor);
dstRow[j][1] = clampToUchar(g * factor);
dstRow[j][2] = clampToUchar(r * factor);
```

---

## 18. Mental Model Summary

Every filter is a rule:

```text
old pixel -> new pixel
```

More advanced filters expand the input:

```text
old pixel + position -> new pixel
old pixel + neighbors -> new pixel
old pixel + face/depth mask -> new pixel
```

Once you understand that, filters become much less mysterious. You are just designing a transformation rule for every pixel in the image.

