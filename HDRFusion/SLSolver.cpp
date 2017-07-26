/*
 * SLSolver.cpp
 *
 *  Created on: Jul 26, 2017
 *      Author: luca
 */

#include "SLSolver.h"

SLSolver::SLSolver(Mat A, Mat b){
	this->A = A;
	this->b = b;

	this->LUFatt();
}

void SLSolver::LUFatt(){
	for(int k = 0; k < A.cols; k++){
		for(int i = k + 1; i < A.cols; i++){
			A.at<float>(i, k) = (float) ((float)A.at<float>(i, k) / A.at<float>(k, k));
			for(int j = k + 1; j < A.cols; j++){
				A.at<float>(i, j) = (float) (A.at<float>(i,j) - A.at<float>(i,k) * A.at<float>(k,j));
			}
		}
	}

	L = Mat::eye(A.cols, A.cols, CV_32F);
	for(int j = 0; j < A.cols; j++){
			for(int i = j + 1; i < A.cols; i++){
				L.at<float>(i,j) = (float) A.at<float>(i,j);
			}
		}

	U = Mat::zeros(A.cols, A.cols, CV_32F);
	for(int i = 0; i < A.cols; i++){
			for(int j = i; j < A.cols; j++){
				U.at<float>(i,j) = (float) A.at<float>(i,j);
			}
		}
}

Mat SLSolver::back_sostitution(Mat U, Mat b){
	Mat x = Mat::zeros(U.cols, 1, CV_32F);
	float sum = 0;
	for (int i = U.cols; i >= 0; i--){
	    sum = 0;
	    for (int j = i + 1; j < U.cols; j++){
	        sum += (float) U.at<float>(i,j) * x.at<float>(j,0);
	    }
	    x.at<float>(i, 0)= (float) ((float)b.at<float>(i, 0) - sum) / (float)U.at<float>(i,i);
	}
	return x.clone();
}

Mat SLSolver::forward_sostitution(Mat L, Mat b){
	Mat x = Mat::zeros(L.cols, 1, CV_32F);
	float sum = 0;
	for (int i = 0; i < L.cols; i++){
	    sum = 0;
	    for(int j = 0; j < i ; j++){
	        sum += (float) L.at<float>(i,j) * x.at<float>(j, 0);
	    }
	    x.at<float>(i, 0)= (float) ((float)b.at<float>(i, 0) - sum) / (float)L.at<float>(i,i);
	}
	return x.clone();
}

Mat SLSolver::solve(){
	Mat y = this->forward_sostitution(this->L, this->b);
	Mat x = this->back_sostitution(this->U, y);
	return x.clone();
}
