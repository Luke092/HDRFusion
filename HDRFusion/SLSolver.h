/*
 * SLSolver.h
 *
 *  Created on: Jul 26, 2017
 *      Author: luca
 */

#ifndef HDRFUSION_HDRFUSION_SLSOLVER_H_
#define HDRFUSION_HDRFUSION_SLSOLVER_H_

#include <core.hpp>

#include <iostream>

using namespace std;
using namespace cv;

class SLSolver {
private:
	Mat L, U;
	Mat A;
	Mat b;

	void LUFatt();
	Mat back_sostitution(Mat U, Mat b);
	Mat forward_sostitution(Mat L, Mat b);
public:
	SLSolver(Mat A, Mat b);
	Mat solve();
};

#endif /* HDRFUSION_HDRFUSION_SLSOLVER_H_ */
