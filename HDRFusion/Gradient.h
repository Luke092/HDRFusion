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
	Mat Gx1;
	Mat Gy1;
	Mat Vx1;
	Mat Vy1;
	int l,N;
	Mat Avg;
	void updateAvg();
	void updateGradient();
	void generateDivG();
public:
	Mat divG;
	Mat U;
	Mat result;
	Gradient(ImageTensor G);
	virtual ~Gradient();
	void update();
	void poissonSolver();
	void poissonSolverGS(int error);
	void addColor(vector<Mat> stack);
};

#endif /* HDRFUSION_HDRFUSION_GRADIENT_H_ */
