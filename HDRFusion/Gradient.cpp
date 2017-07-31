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
	//transpose(Avg, AvgT);
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
//	int Imax = Gx.rows + 2;
//	int Jmax = Gx.cols + 2;

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

//	namedWindow("divG", WINDOW_AUTOSIZE);
//	Mat divGNorm;
//	normalize(divG, divGNorm, 1, 0, NORM_MINMAX);
//	imshow("divG", divGNorm);
//	waitKey(0);
}


//void Gradient::poissonSolver()
//{
//	/*
//	 * (1) Compute Bbar = inv(Q)*B*Q
//  	   (2) For each j,k compute
//         	 Ubar(j,k) = Bbar(j,k)/(Lambda(j,j)+Lambda(k,k))
//  	   (3) Compute U = Q*Ubar*inv(Q)
//	 */
//
//	generateDivG();
//
//	// divG.convertTo(divG, CV_64F);
//
////	normalize(divG, divG, 0, 1, CV_MINMAX);
//
//	/*
//	 *  % Solve the discrete Poisson equation
//		% on an n-by-n grid with right hand side b
//		function X=Poisson_FFT(B)
//		[n,m]=size(b);
//		% Form eigenvalues of matrix T(nxn)
//		L=2*(1-cos((1:n)*pi/(n+1)));
//		% Form reciprocal sums of eigenvalues
//		% Include scale factor 2/(n+1)
//		LL=(2/(n+1))*ones(n,n)./(L'*ones(1,n)+ones(n,1)*L);
//		% Solve, using the Fast Sine Transform
//		X = fast_sine_transform(b');
//		X = fast_sine_transform(X');
//		X = LL.*X;
//		X = fast_sine_transform(X');
//		X = fast_sine_transform(X');
//	 */
//
//	int n = divG.rows;
//	int m = divG.cols;
//
//	Mat L(1, n, CV_64F);
//	for(int i = 0; i < n; i++)
//		L.at<double>(0,i) =(double) 2*(1-cos(i*M_PI/(n+1)));
//
//
//	Mat LL = Mat::ones(n,n, CV_64F);
//	for(int i = 0; i < n; i++)
//	{
//		for(int j = 0; j < n; j++)
//		{
//			LL.at<double>(i,j) =(double) (2.0f/(n+1)) /(L.at<double>(0,i) + L.at<double>(0,j));
//		}
//	}
//
//	Mat divGT = divG.t();
//
//	Mat X = fastSineTransform(divGT);
//	Mat X2 = X.t();
//	X = fastSineTransform(X2);
//
//	for(int i = 0; i < n; i++)
//	{
//		for(int j = 0; j < n; j++)
//		{
//			X.at<double>(i,j) *= LL.at<double>(i,j);
//		}
//	}
//
////	normalize(X, X, 0, 1, NORM_MINMAX);
//	cout <<"prima" << X << endl;
//	namedWindow("image prima", WINDOW_AUTOSIZE);
//	imshow("image prima", X );
//	waitKey(0);
//	X2 = X.t();
//	X = fastSineTransform(X2);
//	X2 = X.t();
//	X = fastSineTransform(X2);
//
//
//	normalize(X, X, 0, 1, NORM_MINMAX);
//	cout <<"dopo" << X << endl;
//	namedWindow("image", WINDOW_AUTOSIZE);
//	imshow("image", X );
//	waitKey(0);
//
////	Mat padded;                            //expand input image to optimal size
////	int m = getOptimalDFTSize(divG.rows);
////	int n = getOptimalDFTSize(divG.cols); // on the border add zero values
////	copyMakeBorder(divG, padded, 0, m - divG.rows, 0, n - divG.cols, BORDER_CONSTANT, Scalar::all(0));
////	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
////	Mat complexI;
////	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
////
////	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);            // this way the result may fit in the source matrix
////
////	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
////	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
////	Mat magI = planes[0];
//
////	magI += Scalar::all(1);                    // switch to logarithmic scale
////	log(magI, magI);
//
////	// crop the spectrum, if it has an odd number of rows or columns
////	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
////
////	// rearrange the quadrants of Fourier image  so that the origin is at the image center
////	int cx = magI.cols/2;
////	int cy = magI.rows/2;
////
////	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
////	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
////	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
////	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
////
////	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
////	q0.copyTo(tmp);
////	q3.copyTo(q0);
////	tmp.copyTo(q3);
////
////	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
////	q2.copyTo(q1);
////	tmp.copyTo(q2);
//
////	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
////											// viewable image form (float between values 0 and 1).
////
////	normalize(divG, divG, 0, 1, NORM_MINMAX);
////	namedWindow("Input Image", WINDOW_KEEPRATIO);
////	namedWindow("spectrum magnitude", WINDOW_KEEPRATIO);
////	imshow("Input Image"       , divG  );    // Show the result
////	imshow("spectrum magnitude", magI);
////	waitKey();
//}
//
//Mat Gradient::fastSineTransform(Mat v)
//{
//	/*% Fast Sine Transform on input matrix V
//      % of dimension n-by-m
//      function Y=fast_sine_transform(V)
//      [n,m]=size(V);
//      V1=[zeros(1,m);V;zeros(n+1,m)];
//      V2=imag(fft(V1));
//      % In Matlab vectors and matrices are indexed
//      % starting at 1, not 0
//      Y=V2(2:n+1,:);
//	 */
//
//	Mat V1 = Mat::zeros(v.rows+2, v.cols+2,CV_64F);
//	for(int i = 0; i < v.rows; i++)
//	{
//		for(int j = 0; j < v.cols; j++)
//		{
//			V1.at<double>(i+1,j+1) = v.at<double>(i, j);
//		}
//	}
//
////	for(int j = 0; j < V1.cols; j++)
////	{
////		Mat col = V1.col(j);
////		Mat planes[] = {col, Mat::zeros(col.size(), CV_64F)};
////		Mat complexI;
////		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
////
////		dft(complexI, complexI);            // this way the result may fit in the source matrix
////
////		split(complexI, planes);
////
////		Mat newCol = planes[1];
////
////		for(int i = 0; i < V1.rows; i++)
////			V1.at<double>(i,j) = newCol.at<double>(i,0);
////	}
//
////	Mat padded;                            //expand input image to optimal size
////	int m = getOptimalDFTSize(v.rows);
////	int n = getOptimalDFTSize(v.cols); // on the border add zero values
////	copyMakeBorder(v, padded, 0, m - v.rows, 0, n - v.cols, BORDER_CONSTANT, Scalar::all(0));
//	Mat planes[] = {Mat_<double>(V1), Mat::zeros(V1.size(), CV_64F)};
//	Mat complexI;
//	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//
//	dft(complexI, complexI);            // this way the result may fit in the source matrix
//
//	split(complexI, planes);
//
//	Mat V2 = planes[1];
//	int point[2] = {1,1};
//	Mat res = crop(V1, point, V1.rows, V1.cols);
////	normalize(res, res, 0, 1, NORM_MINMAX);
//	// cout << res << endl;
//	return res;
//}

void Gradient::dst(double *gtest, double *gfinal,int h,int w)
{
	int k,r,z;
	unsigned long int idx;

	Mat temp = Mat(2*h+2,1,CV_32F);
	Mat res  = Mat(h,1,CV_32F);

	int p=0;
	for(int i=0;i<w;i++)
	{
		temp.at<float>(0,0) = 0.0;

		for(int j=0,r=1;j<h;j++,r++)
		{
			idx = j*w+i;
			temp.at<float>(r,0) = gtest[idx];
		}

		temp.at<float>(h+1,0)=0.0;

		for(int j=h-1, r=h+2;j>=0;j--,r++)
		{
			idx = j*w+i;
			temp.at<float>(r,0) = -1*gtest[idx];
		}

		Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

		Mat complex1;
		merge(planes, 2, complex1);

		dft(complex1,complex1,0,0);

		Mat planes1[] = {Mat::zeros(complex1.size(), CV_32F), Mat::zeros(complex1.size(), CV_32F)};

		// planes1[0] = Re(DFT(I)), planes1[1] = Im(DFT(I))
		split(complex1, planes1);

		std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

		double fac = -2*imag(two_i);

		for(int c=1,z=0;c<h+1;c++,z++)
		{
			res.at<float>(z,0) = planes1[1].at<float>(c,0)/fac;
		}

		for(int q=0,z=0;q<h;q++,z++)
		{
			idx = q*w+p;
			gfinal[idx] =  res.at<float>(z,0);
		}
		p++;
	}

}

void Gradient::idst(double *gtest, double *gfinal,int h,int w)
{
	int nn = h+1;
	unsigned long int idx;
	dst(gtest,gfinal,h,w);
	for(int  i= 0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			idx = i*w + j;
			gfinal[idx] = (double) (2*gfinal[idx])/nn;
		}

}

void Gradient::transpose(double *mat, double *mat_t,int h,int w)
{

	Mat tmp = Mat(h,w,CV_32FC1);
	int p =0;
	unsigned long int idx;
	for(int i = 0 ; i < h;i++)
	{
		for(int j = 0 ; j < w; j++)
		{

			idx = i*(w) + j;
			tmp.at<float>(i,j) = mat[idx];
		}
	}
	Mat tmp_t = tmp.t();

	for(int i = 0;i < tmp_t.size().height; i++)
		for(int j=0;j<tmp_t.size().width;j++)
		{
			idx = i*tmp_t.size().width + j;
			mat_t[idx] = tmp_t.at<float>(i,j);
		}

}

void Gradient::poissonSolver(){
	generateDivG();

	int w = this->divG.cols;
	int h = this->divG.rows;

	double *gtest = new double[h*w];

	int idx = 0;
	for (int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			idx = i*w + j;
			gtest[idx] = divG.at<float>(i,j);
		}
	}

	double *gfinal = new double[(h-2)*(w-2)];
	double *gfinal_t = new double[(h-2)*(w-2)];
	double *denom = new double[(h-2)*(w-2)];
	double *f3 = new double[(h-2)*(w-2)];
	double *f3_t = new double[(h-2)*(w-2)];
	double *img_d = new double[(h)*(w)];

	dst(gtest,gfinal,h-2,w-2);

	transpose(gfinal,gfinal_t,h-2,w-2);

	dst(gfinal_t,gfinal,w-2,h-2);

	transpose(gfinal,gfinal_t,w-2,h-2);

	int cx=1;
	int cy=1;
	idx = 0;

	for(int i = 0 ; i < w-2;i++,cy++)
	{
		for(int j = 0,cx = 1; j < h-2; j++,cx++)
		{
			idx = j*(w-2) + i;
			denom[idx] = (float) 2*cos(M_PI*cy/( (double) (w-1))) - 2 + 2*cos(M_PI*cx/((double) (h-1))) - 2;

		}
	}

	for(idx = 0 ; idx < (w-2)*(h-2) ;idx++)
	{
		gfinal_t[idx] = gfinal_t[idx]/denom[idx];
	}


	idst(gfinal_t,f3,h-2,w-2);

	transpose(f3,f3_t,h-2,w-2);

	idst(f3_t,f3,w-2,h-2);

	transpose(f3,f3_t,w-2,h-2);

	result = Mat(h,w,CV_32F);

	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			idx = i*w+j;
			result.at<float>(i,j) = (float) f3_t[idx];
		}
	}

	namedWindow("U", WINDOW_AUTOSIZE);
	normalize(result, result, 0, 1, NORM_MINMAX);
	imshow("U", result);
	waitKey(0);
	waitKey(0);
}
