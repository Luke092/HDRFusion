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
#include <imgproc.hpp>
#include <highgui.hpp>
#include <iostream>

#include "Filter.h"
#include "ImageTensor.h"
#include "Utilities.h"

using namespace std;
using namespace cv;

class Gradient
{
private:
	Mat Gx;
	Mat Gy;
	int l,N;
	Mat Avg;
	Mat divG;
	Mat result;
	void updateAvg();
	void updateGradient();
	void generateDivG();
public:
	Gradient(ImageTensor G);
	virtual ~Gradient();
	void update();
	void poissonSolver();
	void poissonSolverGS();
};

#endif /* HDRFUSION_HDRFUSION_GRADIENT_H_ */
