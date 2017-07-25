/*
 * ImageTensor.h
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#ifndef HDRFUSION_HDRFUSION_IMAGETENSOR_H_
#define HDRFUSION_HDRFUSION_IMAGETENSOR_H_

#include "Derivate.h"
#include "Tensor.h"

#include <vector>

using namespace std;

class ImageTensor
{
private:
	vector<vector<Tensor>> G;
public:
	ImageTensor(vector<Derivate> I);
	virtual ~ImageTensor();
	Tensor getTensor(int x, int y);
	int getH();
	int getW();
};

#endif /* HDRFUSION_HDRFUSION_IMAGETENSOR_H_ */
