/*
 * Gradient.cpp
 *
 *  Created on: 25 lug 2017
 *      Author: gianmaria
 */

#include "Gradient.h"

Gradient::Gradient(ImageTensor G)
{
	l = 0;
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
			Avg.at<float>(y,x) = (float) 1.0f / (N*3);
		}
	}
}

void Gradient::updateGradient()
{
	Filter fx(Avg);
	Mat V2x = fx.apply(fx.apply(Gx));

	Mat AvgT;
	AvgT = Avg.t();
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

			float gx = _v2x * expf(-abs(_v2x) / (1 + abs(_v2x)));
			float gy = _v2y * expf(-abs(_v2y) / (1 + abs(_v2y)));

			Gx.at<float>(y,x) = (float) gx;
			Gy.at<float>(y,x) = (float) gy;
		}
	}
}

void Gradient::update(){
	do{
		l++;
		N = pow(2,l) + 1;
		/*this->updateAvg();
		this->updateGradient();
		Mat GxNorm, GyNorm;
		GxNorm = Mat(Gx.rows, Gx.cols, CV_32FC1);
		GyNorm = Mat(Gy.rows, Gy.cols, CV_32FC1);
		normalize(Gx, GxNorm, 1, 0, NORM_MINMAX);
		normalize(Gy, GyNorm, 1, 0, NORM_MINMAX);
		namedWindow("Gx", WINDOW_AUTOSIZE);
		namedWindow("Gy", WINDOW_AUTOSIZE);
		imshow("Gx", GxNorm);
		imshow("Gy", GyNorm);
		waitKey(0);

		Gx = GxNorm.clone();
		Gy = GyNorm.clone();*/

	} while(N < Gx.cols || N < Gy.rows);
	//while(l < 12);

}

void Gradient::generateDivG()
{
	divG = Mat::zeros(Gx.rows, Gx.cols, CV_32F);

	for(int i = 0; i < divG.rows; i++)
	{
		for(int j = 0; j < divG.cols; j++)
		{
			float a,b,c,d;

			a = (float) Gx.at<float>(i,j);
			c = (float) Gy.at<float>(i,j);

			if( j - 1 >= 0)
				b = Gx.at<float>(i,j-1);
			else
				b = 0.0f;

			if( i - 1 >= 0)
				d = Gy.at<float>(i-1,j);
			else
				d = 0.0f;

			divG.at<float>(i,j) = a - b + c - d;
		}
	}
}

void Gradient::poissonSolver(){
	// Jacobi's method
	generateDivG();

	U = Mat::zeros(divG.rows, divG.cols, CV_32F);
	Mat U1 = Mat::zeros(divG.rows, divG.cols, CV_32F);

	Mat div = Mat::zeros(divG.rows, divG.cols, CV_32F);
	namedWindow("divG", WINDOW_AUTOSIZE);
	normalize(divG, div, 0, 1, NORM_MINMAX);
	imshow("divG", div);
	waitKey(0);

	double pErr = std::numeric_limits<double>::infinity();
	double err = std::numeric_limits<double>::infinity();
	do{
		for(int i = 0; i < U.rows; i++){
			for(int j = 0; j < U.cols; j++){
				float u_1i_j;
				float u_i1_j;
				float u_i_1j;
				float u_i_j1;
				if(i != 0)
					u_1i_j = (float) U.at<float>(i - 1,j);
				else
					u_1i_j = 0.0f;
				if(i != U.rows - 1)
					u_i1_j = (float) U.at<float>(i + 1,j);
				else
					u_i1_j = 0.0f;
				if(j != 0)
					u_i_1j = (float) U.at<float>(i,j - 1);
				else
					u_i_1j = 0.0f;
				if(j != U.cols - 1)
					u_i_j1 = (float) U.at<float>(i,j + 1);
				else
					u_i_j1 = 0.0f;

				float b_ij = (float) divG.at<float>(i,j);

				U1.at<float>(i,j) = (float) (u_1i_j + u_i1_j + u_i_1j + u_i_j1 + b_ij) / 4;
			}
		}

		pErr = err;
		err = 0;

		for(int i = 0; i < U.rows; i++){
			for(int j = 0; j < U.cols; j++){
				float u_1i_j;
				float u_i1_j;
				float u_i_1j;
				float u_i_j1;
				if(i != 0)
					u_1i_j = (float) -1 * U1.at<float>(i - 1,j);
				else
					u_1i_j = 0.0f;
				if(i != U.rows - 1)
					u_i1_j = (float) -1 * U1.at<float>(i + 1,j);
				else
					u_i1_j = 0.0f;
				if(j != 0)
					u_i_1j = (float) -1 * U1.at<float>(i,j - 1);
				else
					u_i_1j = 0.0f;
				if(j != U.cols - 1)
					u_i_j1 = (float) -1 * U1.at<float>(i,j + 1);
				else
					u_i_j1 = 0.0f;

				float u_i_j = (float) 4 * U1.at<float>(i,j);

				float real, estimate;
				estimate = u_i_j +  u_1i_j + u_i1_j + u_i_1j + u_i_j1;
				real = (float) divG.at<float>(i,j);
				err += pow(real - estimate, 2);
			}
		}

		err = sqrtf(err);

		U = U1.clone();
		U1 = Mat::zeros(U.size(), CV_32F);

		cout << "Error: " << pErr - err << endl;
	}while (pErr - err > pow(10,-4));
	cout << "Error: " << pErr - err << endl;

	namedWindow("U", WINDOW_KEEPRATIO);
	normalize(U, U, 0, 1, NORM_MINMAX);
	imshow("U", U);
	waitKey(0);
	waitKey(0);
}

void Gradient::poissonSolverGS(){
	// Gauss-Seidel method
	generateDivG();

	Mat div = Mat::zeros(divG.rows, divG.cols, CV_32F);
	namedWindow("divG", WINDOW_AUTOSIZE);
	normalize(divG, div, 0, 1, NORM_MINMAX);
	imshow("divG", div);
	waitKey(0);

	U = Mat::zeros(divG.rows, divG.cols, CV_32F);
	Mat U1 = Mat::zeros(divG.rows, divG.cols, CV_32F);

	double pErr = std::numeric_limits<double>::infinity();
	double err = std::numeric_limits<double>::infinity();
	do{
		for(int i = 0; i < U.rows; i++){
			for(int j = 0; j < U.cols; j++){
				float u_1i_j;
				float u_i1_j;
				float u_i_1j;
				float u_i_j1;
				if(i != 0)
					u_1i_j = (float) U1.at<float>(i - 1,j);
				else
					u_1i_j = 0.0f;
				if(i != U.rows - 1)
					u_i1_j = (float) U.at<float>(i + 1,j);
				else
					u_i1_j = 0.0f;
				if(j != 0)
					u_i_1j = (float) U1.at<float>(i,j - 1);
				else
					u_i_1j = 0.0f;
				if(j != U.cols - 1)
					u_i_j1 = (float) U.at<float>(i,j + 1);
				else
					u_i_j1 = 0.0f;

				float b_ij = (float) divG.at<float>(i,j);

				U1.at<float>(i,j) = (float) (u_1i_j + u_i1_j + u_i_1j + u_i_j1 + b_ij) / 4;
			}
		}

		pErr = err;
		err = 0;

		for(int i = 0; i < U.rows; i++){
			for(int j = 0; j < U.cols; j++){
				float u_1i_j;
				float u_i1_j;
				float u_i_1j;
				float u_i_j1;
				if(i != 0)
					u_1i_j = (float) -1 * U1.at<float>(i - 1,j);
				else
					u_1i_j = 0.0f;
				if(i != U.rows - 1)
					u_i1_j = (float) -1 * U1.at<float>(i + 1,j);
				else
					u_i1_j = 0.0f;
				if(j != 0)
					u_i_1j = (float) -1 * U1.at<float>(i,j - 1);
				else
					u_i_1j = 0.0f;
				if(j != U.cols - 1)
					u_i_j1 = (float) -1 * U1.at<float>(i,j + 1);
				else
					u_i_j1 = 0.0f;

				float u_i_j = (float) 4 * U1.at<float>(i,j);

				float real, estimate;
				estimate = u_i_j +  u_1i_j + u_i1_j + u_i_1j + u_i_j1;
				real = (float) divG.at<float>(i,j);
				err += pow(real - estimate, 2);
			}
		}

		err = sqrtf(err);

		U = U1.clone();
		U1 = Mat::zeros(U.size(), CV_32F);

		cout << "Error: " << pErr - err << endl;
	}while (pErr - err > powf(10, -5));
	cout << "Error: " << pErr - err << endl;

	namedWindow("U", WINDOW_KEEPRATIO);
	normalize(U, U, 0, 1, NORM_MINMAX);
	imshow("U", U);
	waitKey(0);
	waitKey(0);
}

void Gradient::addColor(vector<Mat> stack)
{
	result = Mat::zeros(U.rows, U.cols, CV_32FC3);
	//Mat partial = Mat::zeros(U.rows, U.cols, CV_32FC1);
	vector <Mat> channel1,channel2,channel3;

	float epsilon = 0.01;
	float beta = 0.8;

	for(int k = 0; k < stack.size(); k++)
	{
		stack.at(k).convertTo(stack.at(k), CV_32FC3);
	}

	for(int i = 0; i < U.rows; i++)
	{
		for(int j = 0; j < U.cols; j++)
		{
			Vec3f pixel = result.at<Vec3f>(i,j);

			float num;
			float denom;
			float C[3];
			float Cin;
			float weight;
			float fIn = 0.0f;

			for(int c = 0; c < 3; c++)
			{
				num = 0.0f;
				denom = 0.0f;
				C[c] = 0.0f;

				for(int k = 0; k < stack.size(); k++)
				{
					weight = 0.0f;
					Cin = stack.at(k).at<Vec3f>(i,j).val[c];

					if( Cin > 0.5)
						weight = 1 - Cin + epsilon;
					else
						weight = Cin + epsilon;

					num += weight * Cin;
					denom += weight;
				}

				C[c] = num / denom;
			}
			fIn = (C[0] + C[1] + C[2]) / 3;

			for(int c = 0; c < 3; c++)
			{
				float temp = C[c] / fIn;
				pixel.val[c] = pow(temp, beta) * U.at<float>(i,j);
			}

			result.at<Vec3f>(i,j) = pixel;
		}
	}

	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", result);
	waitKey(0);
	waitKey(0);
}
