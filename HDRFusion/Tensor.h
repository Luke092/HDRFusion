/*
 * Tensor.h
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#ifndef HDRFUSION_HDRFUSION_TENSOR_H_
#define HDRFUSION_HDRFUSION_TENSOR_H_

#include <math.h>
#include <vector>

#include "Derivate.h"

using namespace std;

class Tensor
{
private:
	float g[2][2] = {{0,0},{0,0}};
	int x;
	int y;
	float Ix;
	float Iy;
	float S;
	float E;
	void calcS();
	void calcE();

public:
	Tensor(vector<Derivate> I, int x, int y);
	virtual ~Tensor();

	float getg11();
	float getg12();
	float getg21();
	float getg22();

	float getVx();
	float getVy();
};

#endif /* HDRFUSION_HDRFUSION_TENSOR_H_ */
