/*
 * filter.h
 *
 *  Created on: Jul 24, 2017
 *      Author: luca
 */

#ifndef HDRFUSION_HDRFUSION_FILTER_H_
#define HDRFUSION_HDRFUSION_FILTER_H_

#include <core.hpp>
#include <imgproc.hpp>
#include <highgui.hpp>

using namespace std;
using namespace cv;

class Filter{
public:
    Filter(Mat kernel);
    Mat apply(const Mat img);
    static Mat Gx_kernel();
    static Mat Gy_kernel();

private:
    Mat kernel;
};

#endif /* HDRFUSION_HDRFUSION_FILTER_H_ */
