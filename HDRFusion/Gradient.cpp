/*
 * Gradient.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: luca
 */

#include "Gradient.h"

Gradient::Gradient(Mat img){
	Filter fx(Filter::Gx_kernel());
	Filter fy(Filter::Gy_kernel());
	this->Ix = fx.apply(img);
	this->Iy = fy.apply(img);
}

Mat Gradient::get_Ix(){
	return this->Ix;
}

Mat Gradient::get_Iy(){
	return this->Iy;
}
