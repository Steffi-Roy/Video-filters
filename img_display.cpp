/*Creates a window and displays the image
Quit the window when "q" is pressed
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv){
const char* imageName = argc >=2 ? argv[1] : "pathfinder.png";
Mat src;
src = imread(cv::samples::findFile (imageName));
if(src.empty())
{
    printf("Error opening image");
    return EXIT_FAILURE;
}
namedWindow("Display window", 1);
imshow("Display window", src);



while (1){
    char ch = waitKey();
    if (ch=='q'){
        return EXIT_SUCCESS;
    }
}

}