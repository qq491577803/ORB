#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include <math.h>
#include <random>
using namespace std;

struct fastFeat
{
	vector<cv::Point> point1;
	vector<vector<int>> features1;
	vector<int>responseVal;
};

class ORB {
private:
	cv::Mat rgbImage;
	cv::Mat grayImage;
	int fastFeatThd = 20;
	int pcntThreshold = 12;
	bool isNonMaxSupress = true;
	int nonMaxSupThdSqr = 9;
	vector<cv::Point> point1;
	vector<vector<int>> features1;
	fastFeat fastFeat;//to store point ,featureVec,feature response val
public:
	ORB(cv::Mat gray);
	void FastfeaturePointExtract();
	void NonMaximalSupression();
	int comparePixl(int inputPixl, int midPixl, int asbRespSum);
	int comparePixl(int inputPixl, int midPixl);
	void drawPoint(cv::Mat image, vector<cv::Point>);
};

ORB::ORB(cv::Mat image)
{
	rgbImage = image.clone();	
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	FastfeaturePointExtract();
}

void ORB::drawPoint(cv::Mat image, vector<cv::Point>)
{
	for (int xx = 0; xx < fastFeat.point1.size(); xx++)
	{
		cv::circle(image, cv::Point(fastFeat.point1[xx].y, fastFeat.point1[xx].x), 1, cv::Scalar(0, 255, 0), -1);
	}
}

void ORB::NonMaximalSupression()
{
	if (isNonMaxSupress == false)
		return;
	if (fastFeat.point1.size() < 15)
		return;
	int midX, midY,surX,surY;
	int respVal1, respVal2;
	for (int i = 0; i < fastFeat.point1.size()-1; i++)
	{
		midX = fastFeat.point1[i].x;
		midY = fastFeat.point1[i].y;

		for (int j = i + 1; j < fastFeat.point1.size(); j++)
		{
			surX = fastFeat.point1[j].x;
			surY = fastFeat.point1[j].y;
			if (abs(midX - surX)+ abs(midY - surY) < nonMaxSupThdSqr)
			{
				if (fastFeat.responseVal[i] < fastFeat.responseVal[j])
				{
					fastFeat.features1.erase(fastFeat.features1.begin() + i);
					fastFeat.point1.erase(fastFeat.point1.begin() + i);
					NonMaximalSupression();
				}
				else {
					fastFeat.features1.erase(fastFeat.features1.begin() + j);
					fastFeat.point1.erase(fastFeat.point1.begin() + j);
					NonMaximalSupression();
				}
			}
		}
	}
}

int ORB::comparePixl(int midPixl,int inputPixl,int asbRespSum)
{
	int isCore = 0;
	isCore = inputPixl > midPixl + fastFeatThd ? 1 : 0;
	isCore = inputPixl < midPixl - fastFeatThd ? -1 : 0;
	asbRespSum += abs(isCore);
	return isCore;
}

int ORB::comparePixl(int midPixl, int inputPixl)
{
	int isCore = 0;
	isCore = inputPixl > midPixl + fastFeatThd ? 1 : 0;
	isCore = inputPixl < midPixl - fastFeatThd ? -1 : 0;
	return isCore;
}

void ORB::FastfeaturePointExtract()
{
	fastFeat.point1.clear();
	fastFeat.features1.clear();
	fastFeat.responseVal.clear();
	int asbRespSum;
	vector<int> intensity;
	cv::Mat imgBlock;
	int midPixl;
	int isCor1, isCor2, isCor3, isCor4;
	for (int i = 3; i < grayImage.rows - 3; i++)
	{
		for (int j = 3; j < grayImage.cols - 3; j++)
		{
			intensity.clear();
			asbRespSum = 0;
			midPixl = grayImage.at<uchar>(i, j);
			imgBlock = grayImage(cv::Rect(j - 3, i - 3, 7, 7)).clone();
			// preTest :if 1,5,9,13上的点是非角点，直接排除当前block;
			isCor1 = comparePixl(midPixl,imgBlock.at<uchar>(3, 0));
			isCor2 = comparePixl(midPixl, imgBlock.at<uchar>(6, 3));
			isCor3 = comparePixl(midPixl, imgBlock.at<uchar>(3, 6));
			isCor4 = comparePixl(midPixl, imgBlock.at<uchar>(0, 3));
			if (isCor1 == 0 && isCor2 == 0 && isCor3 == 0 && isCor4 == 0)
				continue;
			//Fast feature
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(2, 0), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(1, 1), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(0, 2), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(0, 3), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(0, 4), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(1, 5), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(2, 6), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(3, 6), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(4, 6), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(5, 5), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(6, 4), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(6, 3), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(6, 2), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(5, 1), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(4, 0), asbRespSum));
			intensity.push_back(comparePixl(midPixl, imgBlock.at<uchar>(3, 0), asbRespSum));
			// 若圆上有连续的N个点的亮度大于Ip+T或小于Ip-T，那么像素p被认为是特征点。N通常取12,叫做FAST-12。也可以是9或11
			int isCorLenCnt = 0;
			for (int i = 0; i < intensity.size(); i++)
			{
				if (intensity[i] == 0)
				{
					continue;
				}
				else {				
					isCorLenCnt += 1;
				}
			}
			if (isCorLenCnt > pcntThreshold)
			{
				fastFeat.point1.push_back(cv::Point(i, j));
				fastFeat.features1.push_back(intensity);
				fastFeat.responseVal.push_back(asbRespSum);
			}
		}
	}
	cv::Mat nosupImage = rgbImage.clone();
	drawPoint(nosupImage, point1);	
	cv::imwrite("./nosup.jpg", nosupImage);
	cout << "nosup point size :" << fastFeat.point1.size() << endl;
	NonMaximalSupression();
	cout << "sup point size :" << fastFeat.point1.size() << endl;
	cv::Mat supImage = rgbImage.clone();
	drawPoint(supImage, point1);
	cv::imwrite("./sup.jpg", supImage);
	cv::imshow("nosup", nosupImage);
	cv::imshow("sup", supImage);
	cv::waitKey(0);
}



