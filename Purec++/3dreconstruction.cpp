//C++ Includes
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
typedef uint32_t uint;
//OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
//����� ������������ find_min_bobyqa
#include "dlib/optimization.h"
//LSD
#include "lsd/Includes/lsd.h"
using namespace std;

//���������� LSD ����� � ���� .eps
void write_eps(double * segs, int n, int dim, char * filename, int xsize, int ysize, double width)
{
	FILE * eps;
	int i;

	/* open file */
	if (strcmp(filename, "-") == 0) eps = stdout;
	else eps = fopen(filename, "w");

	/* write EPS header */
	fprintf(eps, "%%!PS-Adobe-3.0 EPSF-3.0\n");
	fprintf(eps, "%%%%BoundingBox: 0 0 %d %d\n", xsize, ysize);
	fprintf(eps, "%%%%Creator: LSD, Line Segment Detector\n");
	fprintf(eps, "%%%%Title: (%s)\n", filename);
	fprintf(eps, "%%%%EndComments\n");

	/* write line segments */
	for (i = 0; i<n; i++)
	{
		fprintf(eps, "newpath %f %f moveto %f %f lineto %f setlinewidth stroke\n",
			segs[i*dim + 0],
			(double)ysize - segs[i*dim + 1],
			segs[i*dim + 2],
			(double)ysize - segs[i*dim + 3],
			width <= 0.0 ? segs[i*dim + 4] : width);
	}

	/* close EPS file */
	fprintf(eps, "showpage\n");
	fprintf(eps, "%%%%EOF\n");
}

//LSD
double* DoLSD(cv::Mat image, int& numLines)
{
	//����������� ����������� ��� ������������ ���������� ��������� LSD
	cv::Mat grayscaleMat(image.size(), CV_8U);
	cv::cvtColor(image, grayscaleMat, CV_BGR2GRAY);
	double* pgm_image;
	pgm_image = (double *)malloc(image.cols * image.rows * sizeof(double));
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			int i = x + (y * image.cols);
			pgm_image[i] = double(grayscaleMat.data[i]);
		}
	}
	//���������� LSD
	int numLinesDetected;
	double* outArray;
	outArray = lsd(&numLinesDetected, pgm_image, image.cols, image.rows);
	numLines = numLinesDetected;
	write_eps(outArray, numLinesDetected, 7, "lsd.eps", image.cols, image.rows, 0);
	return outArray;
}

class PairOfTwoLines {
	public:
		uint FirstIndex;
		uint SecondIndex;
		PairOfTwoLines(uint a, uint b) {
			FirstIndex = a;
			FirstIndex = b;
		}
		PairOfTwoLines() {
			FirstIndex = 0;
			FirstIndex = 0;
		}
};

class LineScore {
public:
	uint goodPoints;
	uint totalPoints;
	uint LineIndex;
	LineScore(uint a, uint b, uint c) {
		goodPoints = a;
		LineIndex = c;
		totalPoints = b;
	}
};

//��� ����������
bool comparator(const LineScore& l, const LineScore& r) {
	if (l.totalPoints != 0 && r.totalPoints != 0) {
		return ((double)l.goodPoints / l.totalPoints) > ((double)r.goodPoints / r.totalPoints);
	}
	return false;
}

//��������� ���. � ��������� �� a �� b
float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

//��������� ������������� ����� � ��������� �� a �� b
uint RandomInt(uint a, uint b) {
	uint output = a + (rand() % (uint)(b - a + 1));
	return output;
}

//���������� �� ������ �� �������� ����������
double RoundTo(double x) {
	int y = floor(x);
	if ((x - y) >= 0.5)
		y++;
	return (double)y;
}

//Intrinsic matrix
cv::Mat K_inv(double f)
{
	double temp[3][3] = { { f, 0, 0 }, { 0, f, 0 }, { 0, 0, 1 } };
	cv::Mat k = cv::Mat(3, 3, CV_64F, temp);
	return k.inv();
}

//���� ����� ����� �� ����������� �����
int FindIndexOfLine(vector<cv::Point3f> vec, cv::Point3f point) {
	auto position = find_if(vec.begin(), vec.end(), [&](const cv::Point3f& a) { 
		return a.x == point.x && a.y == point.y; 
	});
	if (position != vec.end()) {
		int index = (position - vec.begin()) / 2;
		return index;
	}
	return -1;
}

//Cost �������, ������� ������������ ��� ������� ����� alpha � beta
class cost_function
{
	private:
		
	public:
		cv::Mat lineOne;
		cv::Mat lineTwo;
		cv::Mat lineThree;
		cv::Mat lineFour;
		cv::Mat K_inverted;
		cost_function(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h)
		{ 
			lineOne = cv::Mat(b - a);
			lineOne.convertTo(lineOne, CV_64F);
			lineTwo = cv::Mat(d - c);
			lineTwo.convertTo(lineTwo, CV_64F);
			lineThree = cv::Mat(ff - e);
			lineThree.convertTo(lineThree, CV_64F);
			lineFour = cv::Mat(h - g);
			lineFour.convertTo(lineFour, CV_64F);
			K_inverted = K_inv(f);
		}
		double operator() (const dlib::matrix<double, 0, 1> &arg) const 
			{
				double summ = 0;
				
				double r_x_array[3][3] = { { 1, 0, 0 }, { 0, cos(arg(0)), -sin(arg(0)) }, { 0, sin(arg(0)), cos(arg(0)) } };
				cv::Mat R_x = cv::Mat(3, 3, CV_64F, r_x_array);
				double r_y_array[3][3] = { { cos(arg(1)), 0, sin(arg(1)) }, { 0, 1, 0 }, { -sin(arg(1)), 0, cos(arg(1)) } };
				cv::Mat R_y = cv::Mat(3, 3, CV_64F, r_y_array);
				cv::Mat H = R_y*R_x*K_inverted;
				cv::Mat H_t = H.t();

				cv::Mat l_1 = H_t*lineOne;
				cv::normalize(l_1, l_1);
				l_1.resize(2);
				
				cv::Mat l_2 = H_t*lineTwo;
				cv::normalize(l_2, l_2);
				l_2.resize(2);
				
				cv::Mat l_3 = H_t*lineThree;
				cv::normalize(l_3, l_3);
				l_3.resize(2);
				
				cv::Mat l_4 = H_t*lineFour;
				cv::normalize(l_4, l_4);
				l_4.resize(2);

				cv::Mat multiply_one_mat = l_1.t()*l_2;
				cv::Mat multiply_two_mat = l_3.t()*l_4;
				summ += multiply_one_mat.at<double>(0, 0) * multiply_one_mat.at<double>(0, 0) + multiply_two_mat.at<double>(0, 0) * multiply_two_mat.at<double>(0, 0);
				return summ;
			}
};

//����������� cost function ��� ���� ��� �������������� �����
dlib::matrix<double, 0, 1> minimize_C(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h)
{
	dlib::matrix<double, 0, 1> solution(2);
	solution = 0, 0;
	find_min_bobyqa(cost_function(f, a, b, c, d, e, ff, g, h),
		solution,
		5,    // number of interpolation points
		dlib::uniform_matrix<double>(2, 1, -M_PI/2),  // lower bound constraint
		dlib::uniform_matrix<double>(2, 1, M_PI/2),   // upper bound constraint
		M_PI / 10,    // initial trust region radius
		0.001,  // stopping trust region radius
		100    // max number of objective function evaluations
	);
	return solution;
}

//������� ������� ��� ����� ����� �������������� ��� ��������� alpha � beta
uint countInlierScore(dlib::matrix<double, 0, 1> solution, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint score = 0;
	cv::Mat K_inverted = K_inv(f);
	double temp[3][3] = { { 1, 0, 0 }, { 0, cos(solution(0)), -sin(solution(0)) }, { 0, sin(solution(0)), cos(solution(0)) } };
	cv::Mat R_x = cv::Mat(3, 3, CV_64F, temp);
	double temp2[3][3] = { { cos(solution(1)), 0, sin(solution(1)) }, { 0, 1, 0 }, { -sin(solution(1)), 0, cos(solution(1)) } };
	cv::Mat R_y = cv::Mat(3, 3, CV_64F, temp2);
	cv::Mat H = R_y*R_x*K_inverted;
	cv::Mat H_t = H.t();
	for (int i = 0; i < LinePairsVector.size(); i++) {
		PairOfTwoLines pair = LinePairsVector[i];
		cv::Vec3f lineOne = ExtendedLinesVector[2 * pair.FirstIndex + 1] - ExtendedLinesVector[2 * pair.FirstIndex];
		cv::Vec3f lineTwo = ExtendedLinesVector[2 * pair.SecondIndex + 1] - ExtendedLinesVector[2 * pair.SecondIndex];
		
		cv::Mat first_line = cv::Mat(lineOne);
		first_line.convertTo(first_line, CV_64F);
		cv::Mat second_line = cv::Mat(lineTwo);
		second_line.convertTo(second_line, CV_64F);
		
		cv::Mat l_1 = H_t*first_line;
		cv::normalize(l_1, l_1);
		l_1.resize(2);
		
		cv::Mat l_2 = H_t*second_line;
		cv::normalize(l_2, l_2);
		l_2.resize(2);
		
		cv::Mat multiply_one_mat = l_1.t()*l_2;
		double summ = multiply_one_mat.at<double>(0, 0) * multiply_one_mat.at<double>(0, 0);
		if (summ <= threshold) {
			score++;
		}
	}
	return score;
}

//�������� RANSAC (����� ��������� ���� �����, ������ ����������� � ������� ������� ��� ����� ��������������, ����� �������� ������ �������
dlib::matrix<double, 0, 1> RANSAC(uint maxTrials, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint counter = 0;
	uint bestScore = 0;
	dlib::matrix<double, 0, 1> bestSolution(2);
	while (counter < maxTrials) {
		uint first_index = RandomInt(0, LinePairsVector.size() - 1);
		uint second_index = first_index;
		while (second_index == first_index) {
			second_index = RandomInt(0, LinePairsVector.size() - 1);
		}
		PairOfTwoLines pairOne = LinePairsVector[first_index];
		PairOfTwoLines pairTwo = LinePairsVector[second_index];
		cv::Point3f a = ExtendedLinesVector[2 * pairOne.FirstIndex];
		cv::Point3f b = ExtendedLinesVector[2 * pairOne.FirstIndex + 1];
		cv::Point3f c = ExtendedLinesVector[2 * pairOne.SecondIndex];
		cv::Point3f d = ExtendedLinesVector[2 * pairOne.SecondIndex + 1];
		cv::Point3f e = ExtendedLinesVector[2 * pairTwo.FirstIndex];
		cv::Point3f ff = ExtendedLinesVector[2 * pairTwo.FirstIndex + 1];
		cv::Point3f g = ExtendedLinesVector[2 * pairTwo.SecondIndex];
		cv::Point3f h = ExtendedLinesVector[2 * pairTwo.SecondIndex + 1];

		dlib::matrix<double, 0, 1> solution = minimize_C(f, a, b, c, d, e, ff, g, h);
		uint score = countInlierScore(solution, threshold, f, LinePairsVector, ExtendedLinesVector);
		if (score > bestScore) {
			bestScore = score;
			bestSolution = solution;
		}
		counter++;
	}
	cout << "Number of inliers: " << bestScore << endl;
	return bestSolution;
}

void ExtendLines(int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f>& ExtendedLinesVector, vector<PairOfTwoLines>& LinePairsVector, double threshold, int maxX, int maxY) {
	cv::Octree mainTree = cv::Octree(LsdLinesVector);
	for (int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = LsdLinesVector[2 * i];
		cv::Point3f endPoint = LsdLinesVector[2 * i + 1];
		
		//������� ������ ��� �������� �����
		cv::Vec3f currentLine = endPoint - beginPoint;
		double currentLength = cv::norm(currentLine);
		cv::Vec3f normalizedCurrentLine;
		cv::normalize(currentLine, normalizedCurrentLine);
		
		int indexOfLeftIntersection = -1;
		int indexOfRightIntersection = -1;

		vector<cv::Point3f> leftPoints; //� ���� ������� ����� ������ �����, ����������� � Octree � ������� 10 �������� �� ����� ������ �����
		mainTree.getPointsWithinSphere(beginPoint, 10, leftPoints);
		for (int j = 0; j < leftPoints.size(); j++) 
		{
			cv::Point3f newBeginPoint = leftPoints[j];
			if (newBeginPoint != beginPoint) //���������, ��� ����� ����� ������ �����
			{
				cv::Vec3f newLine = endPoint - newBeginPoint;
				double newLength = cv::norm(newLine);
				if (newLength != currentLength) //���� ����� ������, �� ����� �������������
				{
					cv::Vec3f normalizedNewLine;
					cv::normalize(newLine, normalizedNewLine);
					cv::Vec3f difference = normalizedNewLine - normalizedCurrentLine;
					if (cv::norm(difference) <= threshold) //���� ���������� ����, �� ��� �� �� �����, �� ���� ����� ����������
					{
						beginPoint = newBeginPoint;
						int index = FindIndexOfLine(LsdLinesVector, newBeginPoint);
						if (index != -1) {
							indexOfLeftIntersection = index;
						}
						break;
					}
				}
				else 
				{
					cv::Vec3f difference = newLine - currentLine;
					if (cv::norm(difference) <= threshold)
					{
						beginPoint = newBeginPoint;
						int index = FindIndexOfLine(LsdLinesVector, newBeginPoint);
						if (index != -1) {
							indexOfLeftIntersection = index;
						}
						break;
					}
				}
			}
		}

		vector<cv::Point3f> rightPoints; //� ���� ������� ����� ������ �����, ����������� � Octree � ������� 10 �������� �� ����� ����� �����
		mainTree.getPointsWithinSphere(endPoint, 10, rightPoints);
		for (int j = 0; j < rightPoints.size(); j++)
		{
			cv::Point3f newEndPoint = rightPoints[j];
			if (newEndPoint != endPoint)
			{
				cv::Vec3f newLine = newEndPoint - beginPoint;
				double newLength = norm(newLine);
				if (newLength != currentLength)
				{
					cv::Vec3f normalizedNewLine;
					cv::normalize(newLine, normalizedNewLine);
					cv::Vec3f difference = normalizedNewLine - normalizedCurrentLine;
					if (cv::norm(difference) <= threshold)
					{
						endPoint = newEndPoint;
						int index = FindIndexOfLine(LsdLinesVector, newEndPoint);
						if (index != -1) {
							indexOfRightIntersection = index;
						}
						break;
					}
				}
				else
				{
					cv::Vec3f difference = newLine - currentLine;
					if (cv::norm(difference) <= threshold)
					{
						endPoint = newEndPoint;
						int index = FindIndexOfLine(LsdLinesVector, newEndPoint);
						if (index != -1) {
							indexOfRightIntersection = index;
						}
						break;
					}
				}
			}
		}
		
		//��������� ���������� ����� � ��� ������, � ������ ������� �����
		if (endPoint.x >= beginPoint.x) {
			ExtendedLinesVector.push_back(beginPoint);
			ExtendedLinesVector.push_back(endPoint);
		}
		else 
		{
			ExtendedLinesVector.push_back(endPoint);
			ExtendedLinesVector.push_back(beginPoint);
		}

		//���������� � ��� ������������ ���� �����
		if (indexOfLeftIntersection != -1) 
		{
			PairOfTwoLines pair = PairOfTwoLines(i, indexOfLeftIntersection);
			auto pos = find_if(LinePairsVector.begin(), LinePairsVector.end(), [&](const PairOfTwoLines& a){
				return (a.FirstIndex == pair.FirstIndex && a.SecondIndex == pair.SecondIndex) || (a.FirstIndex == pair.SecondIndex && a.SecondIndex == pair.FirstIndex);
			});
			if (pos == LinePairsVector.end()) {
				LinePairsVector.push_back(pair);
			}
		}
		if (indexOfRightIntersection != -1)
		{
			PairOfTwoLines pair = PairOfTwoLines(i, indexOfRightIntersection);
			auto pos = find_if(LinePairsVector.begin(), LinePairsVector.end(), [&](const PairOfTwoLines& a){
				return (a.FirstIndex == pair.FirstIndex && a.SecondIndex == pair.SecondIndex) || (a.FirstIndex == pair.SecondIndex && a.SecondIndex == pair.FirstIndex);
			});
			if (pos == LinePairsVector.end()) {
				LinePairsVector.push_back(pair);
			}
		}
	}
	//��������� ������� ����� ��� �����
	ExtendedLinesVector.push_back(cv::Point3f(0, 0, 0));
	ExtendedLinesVector.push_back(cv::Point3f(0, maxY, 0));

	ExtendedLinesVector.push_back(cv::Point3f(maxX, 0, 0));
	ExtendedLinesVector.push_back(cv::Point3f(maxX, maxY, 0));

	ExtendedLinesVector.push_back(cv::Point3f(0, 0, 0));
	ExtendedLinesVector.push_back(cv::Point3f(maxX, 0, 0));

	ExtendedLinesVector.push_back(cv::Point3f(0, maxY, 0));
	ExtendedLinesVector.push_back(cv::Point3f(maxX, maxY, 0));
}

void getVanishingPoints(double alpha, double beta, double f, vector<cv::Point3f>& output) {

	double kArray[3][3] = { { f, 0, 0 }, { 0, f, 0 }, { 0, 0, 1 } };
	cv::Mat K_matrix = cv::Mat(3, 3, CV_64F, kArray);
	double r_x_array[3][3] = { { 1, 0, 0 }, { 0, cos(alpha), -sin(alpha) }, { 0, sin(alpha), cos(alpha) } };
	cv::Mat Rx = cv::Mat(3, 3, CV_64F, r_x_array);
	double r_y_array[3][3] = { { cos(beta), 0, sin(beta) }, { 0, 1, 0 }, { -sin(beta), 0, cos(beta) } };
	cv::Mat Ry = cv::Mat(3, 3, CV_64F, r_y_array);

	cv::Mat R = Ry*Rx;
	cv::Mat R_t = R.t();

	double XVectorArray[3][1] = { { 0 }, { 0 }, { 1 } };
	cv::Mat XMat = cv::Mat(3, 1, CV_64F, XVectorArray);
	double YVectorArray[3][1] = { { 0 }, { 1 }, { 0 } };
	cv::Mat YMat = cv::Mat(3, 1, CV_64F, YVectorArray);
	double ZVectorArray[3][1] = { { 1 }, { 0 }, { 0 } };
	cv::Mat ZMat = cv::Mat(3, 1, CV_64F, ZVectorArray);

	cv::Mat Nx = R_t*XMat;
	cv::Mat Ny = R_t*YMat;
	cv::Mat Nz = R_t*ZMat;

	cv::Mat Vx_m = K_matrix*Nx;
	cv::Mat Vy_m = K_matrix*Ny;
	cv::Mat Vz_m = K_matrix*Nz;

	cv::Point3f Vx = cv::Point3f(Vx_m);
	cv::Point3f Vy = cv::Point3f(Vy_m);
	cv::Point3f Vz = cv::Point3f(Vz_m);

	output.push_back(Vx);
	output.push_back(Vy);
	output.push_back(Vz);
}

void assignDirections(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<cv::Point3f> VanishingPoints, vector<uint>& output) {
	for (int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		cv::Vec3f line = endPoint - beginPoint;
		cv::Point3f middlePoint = cv::Point3f((endPoint.x - beginPoint.x) / 2, (endPoint.y - beginPoint.y) / 2, 0); //����� �������� �����

		cv::Vec3f lineToX = middlePoint - VanishingPoints[0];
		double cosX = abs(line.dot(lineToX) / (cv::norm(line)*cv::norm(lineToX)));
		cv::Vec3f lineToY = middlePoint - VanishingPoints[1];
		double cosY = abs(line.dot(lineToY) / (cv::norm(line)*cv::norm(lineToY)));
		cv::Vec3f lineToZ = middlePoint - VanishingPoints[2];
		double cosZ = abs(line.dot(lineToZ) / (cv::norm(line)*cv::norm(lineToZ)));

		uint result;
		if (cosX <= 0.9 && cosY <= 0.9 && abs(cosX - cosY) < 0.01) {
			result = 3;
		}
		else if (cosX <= 0.9 && cosZ <= 0.9 && abs(cosX - cosZ) < 0.01) {
			result = 4;
		}
		else if (cosY <= 0.9 && cosZ <= 0.9 && abs(cosY - cosZ) < 0.01) {
			result = 5;
		}
		else if (cosX <= 0.9) {
			result = 0;
		}
		else if (cosY <= 0.9) {
			result = 1;
		}
		else if (cosZ <= 0.9) {
			result = 2;
		}
		output.push_back(result);
	}
}

void getPolygons(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, double AngleTolerance, double step, double radius, vector<vector<uint>>& PolygonsVector) {
	//��������� ����� ������������ �����
	vector<vector<uint>> ParallelLineGroups;
	vector<uint> FirstGroup;
	ParallelLineGroups.push_back(FirstGroup);
	vector<double> GroupCoefficients;
	GroupCoefficients.push_back(0);
	vector<uint> VerticalLinesGroup;
	for (int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		if (endPoint.x - beginPoint.x != 0)
		{
			double AngleCoef = (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x);
			bool Found = false;
			for (int j = 0; j < GroupCoefficients.size(); j++)
			{
				if (abs(AngleCoef - GroupCoefficients[j]) < AngleTolerance)
				{
					ParallelLineGroups[j].push_back(i);
					Found = true;
				}
			}
			if (!Found)
			{
				GroupCoefficients.push_back(AngleCoef);
				vector<uint> Group;
				Group.push_back(i);
				ParallelLineGroups.push_back(Group);
			}
		}
		else
		{
			VerticalLinesGroup.push_back(i);
		}
	}
	ParallelLineGroups.push_back(VerticalLinesGroup);

	vector<LineScore> LineScoreVector;
	for (int i = 0; i < numLinesDetected; i++) {
		LineScore temp = LineScore(0, 0, i);
		LineScoreVector.push_back(temp);
	}
	
	//������������� ����� � ����� step
	vector<cv::Point3f> pointsForSearch;
	vector<uint> PointIndexes;
	for (int i = 0; i < numLinesDetected; i++)
	{
		uint totalPoints = 0;
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		double currentX = beginPoint.x + step;
		while (currentX <= endPoint.x)
		{
			double currentY = (currentX - beginPoint.x) * (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x) + beginPoint.y;
			cv::Point3f point = cv::Point3f(currentX, currentY, 0);
			pointsForSearch.push_back(point);
			PointIndexes.push_back(i); //���������� ����� ����� � ����� ����� �����������
			totalPoints++;
			currentX += step;
		}
		if (totalPoints != 0)
		{
			LineScoreVector[i].totalPoints = totalPoints;
		}
	}

	//����� ����� � Octree
	cv::Octree tree = cv::Octree(pointsForSearch);
	for (int i = 0; i < ParallelLineGroups.size(); i++)
	{
		vector<uint> Group = ParallelLineGroups[i];
		if (Group.size() >= 2)
		{
			for (int j = 0; j < Group.size(); j++)
			{

				for (int k = j + 1; k < Group.size(); k++)
				{

					uint FirstIndex = Group[j];
					uint SecondIndex = Group[k];
					cv::Point3f leftLineBeginPoint;
					cv::Point3f leftLineEndPoint;
					cv::Point3f rightLineBeginPoint;
					cv::Point3f rightLineEndPoint;
					//������� ����� ����� �����
					if (ExtendedLinesVector[2 * SecondIndex + 1].x >= ExtendedLinesVector[2 * FirstIndex + 1].x)
					{
						leftLineBeginPoint = ExtendedLinesVector[2 * FirstIndex];
						leftLineEndPoint = ExtendedLinesVector[2 * FirstIndex + 1];
						rightLineBeginPoint = ExtendedLinesVector[2 * SecondIndex];
						rightLineEndPoint = ExtendedLinesVector[2 * SecondIndex + 1];
					}
					else
					{
						rightLineBeginPoint = ExtendedLinesVector[2 * FirstIndex];
						rightLineEndPoint = ExtendedLinesVector[2 * FirstIndex + 1];
						leftLineBeginPoint = ExtendedLinesVector[2 * SecondIndex];
						leftLineEndPoint = ExtendedLinesVector[2 * SecondIndex + 1];
					}
					
					//����� ����� ��� ����� ������
					vector<cv::Point3f> centersForSearchBegin;
					vector<LineScore> currentBeginScores = LineScoreVector;
					double currentX = leftLineBeginPoint.x + step / 2;
					while (currentX <= rightLineBeginPoint.x) //������������� �������������� �������� � ����� step/2
					{
						double currentY = (currentX - leftLineBeginPoint.x) * (rightLineBeginPoint.y - leftLineBeginPoint.y) / (rightLineBeginPoint.x - leftLineBeginPoint.x) + leftLineBeginPoint.y;
						cv::Point3f point = cv::Point3f(currentX, currentY, 0);
						centersForSearchBegin.push_back(point);
						currentX += step / 2;
					}
					for (int g = 0; g < centersForSearchBegin.size(); g++)
					{
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchBegin[g], radius, foundedPoints);
						for (int h = 0; h < foundedPoints.size(); h++)
						{
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a){ //�������������� � ����� ����� ����������� ��������� �����
								return a.x == tempPoint.x && a.y == tempPoint.y;
							});
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if (index != FirstIndex && index != SecondIndex)
							{

								currentBeginScores[index].goodPoints++;
							}
						}
					}
					sort(currentBeginScores.begin(), currentBeginScores.end(), comparator); //���������
					uint BeginLineIndex = currentBeginScores[0].LineIndex; //�������� ������

					//����� ����� ��� ����� ������
					vector<cv::Point3f> centersForSearchEnd;
					vector<LineScore> currentEndScores = LineScoreVector;
					currentX = leftLineEndPoint.x + step / 2;
					while (currentX <= rightLineEndPoint.x)
					{
						double currentY = (currentX - leftLineEndPoint.x) * (rightLineEndPoint.y - leftLineEndPoint.y) / (rightLineEndPoint.x - leftLineEndPoint.x) + leftLineEndPoint.y;
						cv::Point3f point = cv::Point3f(currentX, currentY, 0);
						centersForSearchEnd.push_back(point);
						currentX += step / 2;
					}
					for (int g = 0; g < centersForSearchEnd.size(); g++)
					{
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchEnd[g], radius, foundedPoints);
						for (int h = 0; h < foundedPoints.size(); h++)
						{
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a){
								return a.x == tempPoint.x && a.y == tempPoint.y;
							});
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if (index != FirstIndex && index != SecondIndex)
							{

								currentEndScores[index].goodPoints++;
							}
						}
					}
					sort(currentEndScores.begin(), currentEndScores.end(), comparator);
					uint EndLineIndex = currentEndScores[0].LineIndex;

					if (currentEndScores[0].goodPoints != 0 && currentBeginScores[0].goodPoints != 0) { //���� ����� ��� �����, �� ��������� ��������� ������� (�� 4-�� �������� �����)
						vector<uint> FoundedPolygon = { FirstIndex, SecondIndex, BeginLineIndex, EndLineIndex };
						PolygonsVector.push_back(FoundedPolygon);
					}

				}
			}
		}
	}
	cout << "Found number of polygons: " << PolygonsVector.size() << endl;
}

double lineSupportingScore(vector<uint> lines, uint lineIndex, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector) {
	double result = 0;
	double summ = 0;
	cv::Point3f CurrentLineBeginPoint = ExtendedLinesVector[lineIndex * 2];
	cv::Point3f CurrentLineEndPoint = ExtendedLinesVector[lineIndex * 2 + 1];
	double currentLineLength = cv::norm(CurrentLineEndPoint - CurrentLineBeginPoint);

	cv::Point3f LSDLineBeginPoint = LsdLinesVector[lineIndex * 2];
	cv::Point3f LSDLineEndPoint = LsdLinesVector[lineIndex * 2 + 1];
	double LSDLineLength = cv::norm(LSDLineEndPoint - LSDLineBeginPoint);

	for (int i = 0; i < lines.size(); i++) {
		cv::Point3f tempPointOne = ExtendedLinesVector[lines[i] * 2];
		cv::Point3f tempPointTwo = ExtendedLinesVector[lines[i] * 2 + 1];
		double length = cv::norm(tempPointTwo - tempPointOne);
		summ += length;
	}
	result = currentLineLength / summ + LSDLineLength / currentLineLength;
	return result;

}

//� ����������
double DirectionScore(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines) {
	double result = 0;
	double Wocc = 1; // �������� ������� ����� ������������
	for (int i = 0; i < lines.size(); i++) {
		uint index = lines[i];
		if (direction == DirectionsOfLines[index]) {
			result += Wocc*lineSupportingScore(lines, index, numLinesDetected, LsdLinesVector, ExtendedLinesVector);
		}
	}
	return result;
}

//� ����������
double dataCost(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines) {
	double result = 0;
	uint Warea = 100; // �������� ������� ����� ������������
	if (direction == 0) {
		result = -Warea*(DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 5, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines));
	}
	else if (direction == 1) {
		result = -Warea*(DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 4, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines));
	}
	else if (direction == 2) {
		result = -Warea*(DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 3, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines));
	}
	return result;
	
}

//� ����������
double EnergyFunction(vector<uint> PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint>> PolygonsVector) {
	double dataCostValue = 0;
	double smoothnessCostValue = 0;

	for (int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> lines = PolygonsVector[i];
		uint normalDirection = PlaneNormals[i];
		dataCostValue += dataCost(lines, normalDirection, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines);
	}
	return dataCostValue + smoothnessCostValue;
}







int main()
{	
	clock_t tStart = clock();
	srand(time(0)); //����� ������������������ ��������� ����� ���� ������ ��������� ��� ����� �������

	//���������� ��� ���������
	double focal_length = 4; //��������� ���������� � mm
	double sensor_width = 4.59; //������ ������� � mm
	double ExtendThreshold = 0.01; //����� ���������� ����� (��� ���������)
	double countInlierThreshold = 0.0001; //���� ������� ���������� ������������ ���� ����� ������ ����� �����, �� �� ������� ��� ����� ��������������
	
	double AngleTolerance = 0.0001; //���� abs(tg(angle1) - tg(angle2)) < AngleTolerance, �� ��� ��� ����� ������������ � ���� ������ (������������ ����� � ��������� �������� �������������)
	double step = 6; //��� ��� ������������� �����
	double radius = 20; //������ ������ ����� ����� �������������� �����
	
	uint ResizeIfMoreThan = 2000; //���� ������ ��� ������ ����������� ������ ����� �����, �� �� ������ ������ �����������
	bool debug = 1;
	
	//�������� �����������
	cv::Mat image;
	image = cv::imread("test.jpg", CV_LOAD_IMAGE_COLOR);
	
	//��������� �������
	uint maxRes = max(image.cols, image.rows);
	if (maxRes > ResizeIfMoreThan)
	{
		float scaleFactor = float(ResizeIfMoreThan) / maxRes;
		cv::Size size = cv::Size(image.cols*scaleFactor, image.rows*scaleFactor);
		cv::resize(image, image, size);
	}
	
	//������� �����
	uint maxX = image.cols;
	uint maxY = image.rows;
	
	//��������� ����������
	double temp_focal = maxX*focal_length / sensor_width;
	uint f = (uint)temp_focal; //��������� ���������� � ��������

	//LSD
	int numLinesDetected;
	double* LsdLinesArray = DoLSD(image, numLinesDetected);
	cout << "Number of LSD lines detected: " << numLinesDetected << endl;

	//�������� ����� �� ������� � ������ � ������ �������
	vector<cv::Point3f> LsdLinesVector; //������� � ������� 2*i ���� ��� ����� ������ i-�� �����, ������� � ������� 2*i + 1 ���� ��� ����� ����� i-�� �����
	for (int i = 0; i < numLinesDetected; i++) 
	{
		double x1 = RoundTo(LsdLinesArray[7 * i]);
		double y1 = RoundTo(LsdLinesArray[7 * i + 1]);
		double x2 = RoundTo(LsdLinesArray[7 * i + 2]);
		double y2 = RoundTo(LsdLinesArray[7 * i + 3]);
		cv::Point3f point_one = cv::Point3f(x1, y1, 0);
		cv::Point3f point_two = cv::Point3f(x2, y2, 0);
		if (x1 >= x2) 
		{
			LsdLinesVector.push_back(point_one);
			LsdLinesVector.push_back(point_two);
		}
		else 
		{
			LsdLinesVector.push_back(point_two);
			LsdLinesVector.push_back(point_one);
		}
	}

	//���������
	vector<cv::Point3f> ExtendedLinesVector; //������� � �������� 2*i ���� ��� ����� ������ i-�� �����, ������� � �������� 2*i + 1 ���� ��� ����� ����� i-�� �����
	vector<PairOfTwoLines> LinePairsVector; //� ������ �������� ����� ������� ����� ��� ������� (�����, ������� ������������)
	ExtendLines(numLinesDetected, LsdLinesVector, ExtendedLinesVector, LinePairsVector, ExtendThreshold, maxX, maxY);
	cout << "Number of line pairs: " << LinePairsVector.size() << endl;
	
	//RANSAC (������� ���� ����� � ����)
	uint maxRansacTrials = (uint)LinePairsVector.size(); //������������ ���������� �������� ��������� RANSAC
	dlib::matrix<double, 0, 1> solution = RANSAC(maxRansacTrials, countInlierThreshold, f, LinePairsVector, ExtendedLinesVector);
	cout << "angle Alpha (in radians): " << solution(0) << endl << "angle Beta (in radians): " << solution(1) << endl;

	//�������� vanishing points
	vector<cv::Point3f> VanishingPoints;
	getVanishingPoints(solution(0), solution(1), f, VanishingPoints);

	//������� ����������� ������ �����
	vector<uint> DirectionsOfLines; //������� � ������� i �������� ����������� i-�� ����� (0=x, 1=y, 2=z, 3=xy, 4=xz, 5=yz)
	assignDirections(numLinesDetected, ExtendedLinesVector, VanishingPoints, DirectionsOfLines);

	//�������� ��� ��������� �������
	vector<vector<uint>> PolygonsVector;
	getPolygons(numLinesDetected, ExtendedLinesVector, AngleTolerance, step, radius, PolygonsVector);

	/*
	//������ � ���� ��� ������
	if (debug) {
		write_eps(LsdLinesArray, numLinesDetected, 7, "lsd_lines.eps", maxX, maxY, 1);
		double * ExtendedLinesArray = (double *)malloc(maxX * maxY * sizeof(double));
		for (int i = 0; i < numLinesDetected; i++) {
			ExtendedLinesArray[7 * i] = ExtendedLinesVector[2 * i].x;
			ExtendedLinesArray[7 * i + 1] = ExtendedLinesVector[2 * i].y;
			ExtendedLinesArray[7 * i + 2] = ExtendedLinesVector[2 * i + 1].x;
			ExtendedLinesArray[7 * i + 3] = ExtendedLinesVector[2 * i + 1].y;
			ExtendedLinesArray[7 * i + 4] = 1;
			ExtendedLinesArray[7 * i + 5] = 0.125;
			ExtendedLinesArray[7 * i + 6] = 15;
		}
		write_eps(ExtendedLinesArray, numLinesDetected, 7, "extended_lines.eps", maxX, maxY, 1);


		for (int i = 0; i < 100; i++) {
			cv::Mat copyImage = image.clone();
			vector<uint> bla = PolygonsVector[i];
			uint index1 = bla[0];
			uint index2 = bla[1];
			uint index3 = bla[2];
			uint index4 = bla[3];
			
			cv::Point3f B_1 = ExtendedLinesVector[index1 * 2];
			cv::Point3f E_1 = ExtendedLinesVector[index1 * 2 + 1];
			cv::Point3f B_2 = ExtendedLinesVector[index2 * 2];
			cv::Point3f E_2 = ExtendedLinesVector[index2 * 2 + 1];
			cv::Point3f B_3 = ExtendedLinesVector[index3 * 2];
			cv::Point3f E_3 = ExtendedLinesVector[index3 * 2 + 1];
			cv::Point3f B_4 = ExtendedLinesVector[index4 * 2];
			cv::Point3f E_4 = ExtendedLinesVector[index4 * 2 + 1];

			vector<cv::Point> tempVec = { cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), cv::Point(B_2.x, B_2.y), cv::Point(E_2.x, E_2.y), cv::Point(B_3.x, B_3.y), cv::Point(E_3.x, E_3.y), cv::Point(B_4.x, B_4.y), cv::Point(E_4.x, E_4.y) };
			
			const cv::Scalar blaColor = cv::Scalar(RandomInt(0, 255), RandomInt(0, 255), RandomInt(0, 255), 255);
			//cv::fillConvexPoly(image, &tempVec[0], 8, blaColor);
			cv::line(copyImage, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 5);
			cv::line(copyImage, cv::Point(B_2.x, B_2.y), cv::Point(E_2.x, E_2.y), blaColor, 5);
			cv::line(copyImage, cv::Point(B_3.x, B_3.y), cv::Point(E_3.x, E_3.y), blaColor, 5);
			cv::line(copyImage, cv::Point(B_4.x, B_4.y), cv::Point(E_4.x, E_4.y), blaColor, 5);
			cv::imwrite("polygons_" + std::to_string(i) + ".jpg", copyImage);
		}
	}*/
	system("pause");
	return 0;
}

