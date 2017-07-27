/*
 * Utilities.cpp
 *
 *  Created on: Jul 27, 2017
 *      Author: luca
 */

#include "Utilities.h"

template <typename PixelType>
Mat template_crop(Mat original, int start_point[], int rows, int cols){
	Mat cropped(rows,cols,original.type());
	for (int y = 0; y < rows; y++)
		for (int x = 0; x < cols; x++){
			int x_old = start_point[0] + x;
			int y_old = start_point[1] + y;
			cropped.at<PixelType>(y,x) = original.at<PixelType>(y_old,x_old);
		}
	return cropped;
}

Mat crop(Mat original, int start_point[], int rows, int cols){
	switch(original.channels()){
	case 1:
		return template_crop<uchar>(original,start_point,rows,cols);
	case 3:
		return template_crop<Vec3b>(original,start_point,rows,cols);
	}
	return Mat();
}


