/*
 * Gradient.h
 *
 *  Created on: Jul 24, 2017
 *      Author: luca
 */

#ifndef HDRFUSION_HDRFUSION_GRADIENT_H_
#define HDRFUSION_HDRFUSION_GRADIENT_H_

#include <core.hpp>

#include "Filter.h"

using namespace cv;

class Gradient {
private:
	Mat Ix;
	Mat Iy;
public:
	Gradient(Mat img);
	Mat get_Ix();
	Mat get_Iy();
};

#endif /* HDRFUSION_HDRFUSION_GRADIENT_H_ */
