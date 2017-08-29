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
}

Mat Derivate::get_Ix(){
	return this->Ix;
}

Mat Derivate::get_Iy(){
	return this->Iy;
}
