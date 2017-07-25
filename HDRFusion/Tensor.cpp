/*
 * Tensor.cpp
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#include "Tensor.h"


template<typename T> int sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}

Tensor::Tensor(vector<Derivate> I, int x, int y)
{
	this->x = x;
	this->y = y;
	this->Ix = 0;
	this->Iy = 0;

	for(unsigned int i = 0; i < I.size(); i++)
	{
		Ix += I[i].get_Ix().at<float>(y,x);
		Iy += I[i].get_Iy().at<float>(y,x);
		g[0][0]+= I[i].get_Ix().at<float>(y,x) * I[i].get_Ix().at<float>(y,x);
		g[0][1]+= I[i].get_Ix().at<float>(y,x) * I[i].get_Iy().at<float>(y,x);
		g[1][0]+= I[i].get_Iy().at<float>(y,x) * I[i].get_Ix().at<float>(y,x);
		g[1][1]+= I[i].get_Iy().at<float>(y,x) * I[i].get_Iy().at<float>(y,x);
	}

//	cout << "g11: " << getg11() <<
//			" g12: " << getg12() <<
//			" g21: " << getg21() <<
//			" g22: " << getg22() << endl;

	calcS();
	calcE();

//	cout << "S: " << S <<
//			" E:" << E << endl;
}

Tensor::~Tensor()
{
	// TODO Auto-generated destructor stub
}

float Tensor::getg11()
{
	return g[0][0];
}

float Tensor::getg12()
{
	return g[0][1];
}

float Tensor::getg21()
{
	return g[1][0];
}

float Tensor::getg22()
{
	return g[1][1];
}

void Tensor::calcS()
{
	S = sqrtf(pow(getg11() - getg22(),2) + 4 * getg12() * getg21());
}

void Tensor::calcE()
{
	E = 0.5 * (getg11() + getg22() + S);
}

float Tensor::getVx()
{
	float Vx = 0;
	float fourg12 = 4 * pow(getg12(),2);
	if(fourg12 == 0)
		Vx = 0;
	else
		Vx = sgn<float>(Ix) * sqrt(E * fourg12 / (fourg12 + pow(getg22() - getg11() + S, 2)));
	return Vx;
}

float Tensor::getVy()
{
	float Vy = 0;
	float g22g11S = pow(getg22() - getg11() + S,2);
	if(g22g11S == 0)
		Vy = 0;
	else
		Vy = sgn<float>(Iy) * sqrtf(E * g22g11S / (4 * pow(getg12(),2) + g22g11S));
	return Vy;
}

