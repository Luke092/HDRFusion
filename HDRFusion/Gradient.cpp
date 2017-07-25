/*
 * Gradient.cpp
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#include "Gradient.h"

Gradient::Gradient(ImageTensor G)
{
	l = 1;
	N = pow(2, l) + 1;
	Gx = Mat(G.getH(), G.getW(), CV_32F);
	Gy = Mat(G.getH(), G.getW(), CV_32F);

	for(int y = 0; y < G.getH(); y++)
	{
		for(int x = 0; x < G.getW(); x++)
		{
			float Vx = G.getTensor(x, y).getVx();
			float Vy = G.getTensor(x, y).getVy();
			Gx.at<float>(y,x) = Vx * expf(-abs(Vx) / (1 + abs(Vx)));
			Gy.at<float>(y,x) = Vy * expf(-abs(Vy) / (1 + abs(Vy)));
		}
	}
	updateAvg();
}

Gradient::~Gradient()
{
	// TODO Auto-generated destructor stub
}

void Gradient::updateAvg()
{
	Avg = Mat(3,N,CV_32F);
	for(int y = 0; y < 3; y++)
	{
		for(int x = 0; x < N; x++)
		{
			Avg.at<float>(y,x) = (float) 1.0f/(3*N);
		}
	}
}

Mat Gradient::updateGradient()
{
	Filter f(Avg);
	Mat V2x = f.apply(f.apply(Gx));
	return V2x;

}

