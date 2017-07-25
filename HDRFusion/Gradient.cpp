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

void Gradient::updateGradient()
{
	Filter fx(Avg);
	Mat V2x = fx.apply(fx.apply(Gx));

	Mat AvgT;
	transpose(Avg, AvgT);
	Filter fy(AvgT);
	Mat V2y = fy.apply(fy.apply(Gy));

	for(int y = 0; y < Gy.rows; y++){
		for(int x = 0; x < Gx.cols; x++){
			float valX = V2x.at<float>(y,x) * (N + 1);
			V2x.at<float>(y,x) = valX;
			float valY = V2y.at<float>(y,x) * (N + 1);
			V2y.at<float>(y,x) = valY;
		}
	}

	for(int y = 0; y < Gy.rows; y++)
	{
		for(int x = 0; x < Gx.cols; x++)
		{
			float _v2x = (float) V2x.at<float>(y,x);
			float _v2y = (float) V2y.at<float>(y,x);
			Gx.at<float>(y,x) = _v2x * expf(-abs(_v2x) / (1 + abs(_v2x)));
			Gy.at<float>(y,x) = _v2y * expf(-abs(_v2y) / (1 + abs(_v2y)));
		}
	}
}

void Gradient::update(){
	do{
		l++;
		N = pow(2,l) + 1;
		this->updateGradient();
		/*Mat GxNorm, GyNorm;
		GxNorm = Mat(Gx.rows, Gx.cols, CV_32FC1);
		GyNorm = Mat(Gy.rows, Gy.cols, CV_32FC1);
		normalize(Gx, GxNorm, 1, 0, NORM_MINMAX);
		normalize(Gy, GyNorm, 1, 0, NORM_MINMAX);
		namedWindow("Gx", WINDOW_AUTOSIZE);
		namedWindow("Gy", WINDOW_AUTOSIZE);
		imshow("Gx", GxNorm);
		imshow("Gy", GyNorm);
		waitKey(0);*/
	} while(N < Gx.cols || N < Gy.rows);

}

