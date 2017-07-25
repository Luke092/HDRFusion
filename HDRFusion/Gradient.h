/*
 * Gradient.h
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#ifndef HDRFUSION_HDRFUSION_GRADIENT_H_
#define HDRFUSION_HDRFUSION_GRADIENT_H_

#include <math.h>
#include <core.hpp>

#include "Filter.h"
#include "ImageTensor.h"


class Gradient
{
private:
	Mat Gx;
	Mat Gy;
	int l,N;
	Mat Avg;
	void updateAvg();
public:
	Gradient(ImageTensor G);
	virtual ~Gradient();
	Mat updateGradient();
};

#endif /* HDRFUSION_HDRFUSION_GRADIENT_H_ */
