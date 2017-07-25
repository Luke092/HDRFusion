/*
 * ImageTensor.cpp
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#include "ImageTensor.h"

ImageTensor::ImageTensor(vector<Derivate> I)
{
	int h = I[0].get_Iy().rows;
	int w = I[0].get_Ix().cols;
	for(int y = 0; y < h; y++)
	{
		G.push_back(vector<Tensor>());
		for(int x = 0; x < w; x++)
		{
			G[y].push_back(Tensor(I, x, y));
		}
	}

}

ImageTensor::~ImageTensor()
{
	// TODO Auto-generated destructor stub
}

Tensor ImageTensor::getTensor(int x, int y)
{
	return G[y][x];
}

int ImageTensor::getH()
{
	return G.size();
}

int ImageTensor::getW()
{
	return G[0].size();
}

