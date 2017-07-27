/*
 * Utilities.h
 *
 *  Created on: Jul 27, 2017
 *      Author: luca
 */

#ifndef HDRFUSION_HDRFUSION_UTILITIES_H_
#define HDRFUSION_HDRFUSION_UTILITIES_H_

#include <core.hpp>
#include <imgproc.hpp>

using namespace cv;

Mat crop(Mat original, int start_point[], int rows, int cols);


#endif /* HDRFUSION_HDRFUSION_UTILITIES_H_ */
