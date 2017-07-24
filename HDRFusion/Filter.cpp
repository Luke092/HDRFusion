/*
 * filter.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: luca
 */

#include "Filter.h"

Filter::Filter(Mat kernel){
	this->kernel = kernel.clone();
}

Mat Filter::apply(const Mat img){
	Mat res(img.rows, img.cols, img.type());
	filter2D(img, res, -1, this->kernel, Point(-1,-1), 0, BORDER_DEFAULT);
	return res.clone();
}

Mat Filter::Gx_kernel(){
	Mat kernelH(1, 3, CV_32F);
	kernelH.at<float>(0,0) = 1.0f;
	kernelH.at<float>(0,1) = 0.0f;
	kernelH.at<float>(0,2) = -1.0f;
	return kernelH.clone();
}

Mat Filter::Gy_kernel(){
	Mat kernelH(3, 1, CV_32F);
	kernelH.at<float>(0,0) = 1.0f;
	kernelH.at<float>(1,0) = 0.0f;
	kernelH.at<float>(2,0) = -1.0f;
	return kernelH.clone();
}
