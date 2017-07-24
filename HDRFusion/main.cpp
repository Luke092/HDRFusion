/* 
 * File:   main.cpp
 * Author: Luca Bettinelli
 *
 * Created on July 18, 2017, 3:20 PM
 */

#include <stdlib.h>
#include <core.hpp>
#include <imgproc.hpp>
#include <highgui.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

/*
 * 
 */

Mat toBN(Mat img);

int main(int argc, char** argv) {
    Mat m = imread("/home/luca/Pictures/lena.jpg", CV_LOAD_IMAGE_COLOR);
    namedWindow("Test", WINDOW_AUTOSIZE);
    imshow("Test", toBN(m));
    waitKey(0);
    return 0;
}

Mat toBN(Mat img){
    Mat res(img.rows, img.cols, CV_8UC1);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            int new_pixel = 0;
            Vec3b pixel = img.at<Vec3b>(i,j);
            for(int k = 0; k < 3; k++){
                new_pixel += (int)pixel.val[k];
            }
            new_pixel = (int)floor(((double)1/3) * new_pixel);
            res.at<uchar>(i,j) = (uchar) new_pixel;
        }   
    }
    return res;
}
