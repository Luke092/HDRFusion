/*
 * Utilities.cpp
 *
 *  Created on: Jul 27, 2017
 *      Author: luca
 */

#include "Utilities.h"

Mat crop(Mat original, int start_point[], int rows, int cols){
	Mat cropped(rows,cols,original.type());
	for (int y = 0; y < rows; y++)
		for (int x = 0; x < cols; x++){
			int x_old = start_point[0] + x;
			int y_old = start_point[1] + y;
			cropped.at<float>(y,x) = original.at<float>(y_old,x_old);
		}
	return cropped;
}


