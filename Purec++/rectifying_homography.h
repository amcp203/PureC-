#pragma once
#include "dlib/optimization.h"
#include <tools.h>
#include <vector>
using namespace std;

cv::Mat K_inv(double f);

//���� ����� ����� �� ����������� �����
int FindIndexOfLine(vector<cv::Point3f> vec, cv::Point3f point);

//Cost �������, ������� ������������ ��� ������� ����� alpha � beta
class cost_function {
private:

public:
	cv::Mat lineOne;
	cv::Mat lineTwo;
	cv::Mat lineThree;
	cv::Mat lineFour;
	cv::Mat K_inverted;

	cost_function(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h);

	double operator()(const dlib::matrix<double, 0, 1>& arg) const;
};

//����������� cost function ��� ���� ��� �������������� �����
dlib::matrix<double, 0, 1> minimize_C(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h);

//������� ������� ��� ����� ����� �������������� ��� ��������� alpha � beta
uint countInlierScore(dlib::matrix<double, 0, 1> solution, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector);

//�������� RANSAC (����� ��������� ���� �����, ������ ����������� � ������� ������� ��� ����� ��������������, ����� �������� ������ �������
dlib::matrix<double, 0, 1> RANSAC_(uint maxTrials, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector);
