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

#include<string.h>
#include<fstream>
#include<dirent.h>

#include "Filter.h"
#include "Gradient.h"

using namespace std;
using namespace cv;

/*
 * 
 */

Mat toBN(Mat img);
vector<string> listFile(const char* dirPath);
bool has_suffix(const string& s, const string& suffix);

int main(int argc, char** argv) {
	vector<string> list = listFile(".");
	vector<Mat> stack;
	vector<Gradient> I;
	for(int i = 0; i < list.size(); i++)
	{
		    Mat m = imread(list[i], CV_LOAD_IMAGE_COLOR);
		    namedWindow(list[i], WINDOW_AUTOSIZE);

		    Mat img = toBN(m);
		    stack.push_back(img);
		    I.push_back(Gradient(img));

		    imshow(list[i], img);
		    waitKey(0);
	}

	for(int i = 0; i < I.size(); i++){
		namedWindow("Ix", WINDOW_AUTOSIZE);
		namedWindow("Iy", WINDOW_AUTOSIZE);
		imshow("Ix", I[i].get_Ix());
		imshow("Iy", I[i].get_Iy());
		waitKey(0);
	}

    return 0;
}

Mat toBN(Mat img){
    Mat res(img.rows, img.cols, CV_32FC1);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            float new_pixel = 0;
            Vec3b pixel = img.at<Vec3b>(i,j);
            for(int k = 0; k < 3; k++){
                new_pixel += (int)pixel.val[k];
            }
            new_pixel = (((double)1/3) * new_pixel)/255;
            res.at<float>(i,j) = (float) new_pixel;
        }   
    }
    return res;
}

bool has_suffix(const string& s, const string& suffix)
{
    return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

vector<string> listFile(const char* dirPath){
        DIR *pDIR;
        vector<string> list;
        struct dirent *entry;
        if( pDIR=opendir(dirPath) ){
                while(entry = readdir(pDIR)){
                	if(has_suffix(entry->d_name, ".jpg"))
                	{
                		string s = entry->d_name;
						list.push_back(s);
                		cout << entry->d_name << "\n";
                	}
                }
                closedir(pDIR);
        }
        return list;
}
