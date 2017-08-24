/*
 * Gradient.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: luca
 */

#include "Derivate.h"

Derivate::Derivate(Mat img){
	Filter fx(Filter::Gx_kernel());
	Filter fy(Filter::Gy_kernel());
	this->Ix = fx.apply(img);
	this->Iy = fy.apply(img);
//	int scale = 1;
//	int delta = 0;
//	Sobel(img,Ix, CV_32F, 1, 0, 7, scale, delta, BORDER_DEFAULT);
//	Sobel(img,Iy, CV_32F, 0, 1, 7, scale, delta, BORDER_DEFAULT);
}

Mat Derivate::get_Ix(){
	return this->Ix;
}

Mat Derivate::get_Iy(){
	return this->Iy;
}
