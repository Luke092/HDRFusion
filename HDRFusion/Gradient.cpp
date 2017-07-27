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

void Gradient::generateDivG()
{
	int Imax = Gx.rows + 2;
	int Jmax = Gx.cols + 2;

	divG = Mat::zeros((Imax), (Jmax), CV_32F);

	for(int i = 0; i < Imax; i++)
	{
		for(int j = 0; j < Jmax; j++)
		{
			if((i != 0) && (j != 0) && (i != Imax-1) && (j != Jmax-1))
			{
				divG.at<float>(i-1, j-1) =(float) Gx.at<float>(i-1,j-1) - Gx.at<float>(i-1, j-2)
									  + Gy.at<float>(i-1,j-1) - Gy.at<float>(i-2,j-1);
			}
		}
	}
	int point[2] = {1,1};
	Mat cropG = crop(divG, point, Gx.rows, Gx.cols);
	divG = cropG;

//	namedWindow("divG", WINDOW_AUTOSIZE);
//	Mat divGNorm;
//	normalize(cropG, divGNorm, 1, 0, NORM_MINMAX);
//	imshow("divG", divGNorm);
//	waitKey(0);
}


void Gradient::poissonSolver()
{
	/*
	 * (1) Compute Bbar = inv(Q)*B*Q
  	   (2) For each j,k compute
         	 Ubar(j,k) = Bbar(j,k)/(Lambda(j,j)+Lambda(k,k))
  	   (3) Compute U = Q*Ubar*inv(Q)
	 */

	generateDivG();

	normalize(divG, divG, 0, 1, CV_MINMAX);

	/*
	 *  % Solve the discrete Poisson equation
		% on an n-by-n grid with right hand side b
		function X=Poisson_FFT(B)
		[n,m]=size(b);
		% Form eigenvalues of matrix T(nxn)
		L=2*(1-cos((1:n)*pi/(n+1)));
		% Form reciprocal sums of eigenvalues
		% Include scale factor 2/(n+1)
		LL=(2/(n+1))*ones(n,n)./(L'*ones(1,n)+ones(n,1)*L);
		% Solve, using the Fast Sine Transform
		X = fast_sine_transform(b');
		X = fast_sine_transform(X');
		X = LL.*X;
		X = fast_sine_transform(X');
		X = fast_sine_transform(X');
	 */

	int n = divG.rows;
	int m = divG.cols;

	Mat L(1, n, CV_32F);
	for(int i = 0; i < n; i++)
		L.at<float>(0,i) =(float) 2*(1-cos(i*M_PI/(n+1)));


	Mat LL = Mat::ones(n,n, CV_32F);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			LL.at<float>(i,j) =(float) (2.0f/(n+1)) /(L.at<float>(0,i) + L.at<float>(0,j));
		}
	}

	Mat divGT = divG.t();

	Mat X = fastSineTransform(divGT);
	Mat X2 = X.t();
	X = fastSineTransform(X2);

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			X.at<float>(i,j) *= LL.at<float>(i,j);
		}
	}

	X2 = X.t();
	X = fastSineTransform(X2);
	X2 = X.t();
	X = fastSineTransform(X2);

	cout << X << endl;

	normalize(X, X, 0, 1, CV_MINMAX);
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", X );
	waitKey(0);

//	Mat padded;                            //expand input image to optimal size
//	int m = getOptimalDFTSize(divG.rows);
//	int n = getOptimalDFTSize(divG.cols); // on the border add zero values
//	copyMakeBorder(divG, padded, 0, m - divG.rows, 0, n - divG.cols, BORDER_CONSTANT, Scalar::all(0));
//	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//	Mat complexI;
//	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//
//	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);            // this way the result may fit in the source matrix
//
//	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//	Mat magI = planes[0];

//	magI += Scalar::all(1);                    // switch to logarithmic scale
//	log(magI, magI);

//	// crop the spectrum, if it has an odd number of rows or columns
//	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//
//	// rearrange the quadrants of Fourier image  so that the origin is at the image center
//	int cx = magI.cols/2;
//	int cy = magI.rows/2;
//
//	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
//	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
//	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
//
//	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//
//	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//	q2.copyTo(q1);
//	tmp.copyTo(q2);

//	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//											// viewable image form (float between values 0 and 1).
//
//	normalize(divG, divG, 0, 1, NORM_MINMAX);
//	namedWindow("Input Image", WINDOW_KEEPRATIO);
//	namedWindow("spectrum magnitude", WINDOW_KEEPRATIO);
//	imshow("Input Image"       , divG  );    // Show the result
//	imshow("spectrum magnitude", magI);
//	waitKey();
}

Mat Gradient::fastSineTransform(Mat v)
{
	/*% Fast Sine Transform on input matrix V
      % of dimension n-by-m
      function Y=fast_sine_transform(V)
      [n,m]=size(V);
      V1=[zeros(1,m);V;zeros(n+1,m)];
      V2=imag(fft(V1));
      % In Matlab vectors and matrices are indexed
      % starting at 1, not 0
      Y=V2(2:n+1,:);
	 */

	Mat V1 = Mat::zeros(v.rows+2, v.cols,CV_32F);
	for(int i = 0; i < v.rows; i++)
	{
		for(int j = 0; j < v.cols; j++)
		{
			V1.at<float>(i+1,j) = v.at<float>(i, j);
		}
	}

//	Mat padded;                            //expand input image to optimal size
//	int m = getOptimalDFTSize(V1.rows);
//	int n = getOptimalDFTSize(V1.cols); // on the border add zero values
//	copyMakeBorder(V1, padded, 0, m - V1.rows, 0, n - V1.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = {Mat_<float>(V1), Mat::zeros(V1.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);            // this way the result may fit in the source matrix

	split(complexI, planes);

	Mat V2 = planes[1];
	int point[2] = {1,1};
	Mat res = crop(V2, point, V2.rows, V2.cols);
	normalize(res, res, 0, 1, NORM_MINMAX);
	// cout << res << endl;
	return res;
}
