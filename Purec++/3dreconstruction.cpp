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
//Чтобы использовать find_min_bobyqa
#include "dlib/optimization.h"
//LSD
#include "lsd/Includes/lsd.h"
using namespace std;

//Записывает LSD линии в файл .eps
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
	//Конвертация изображения для последующего применения алгоритма LSD
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
	//Применение LSD
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

//Для сортировки
bool comparator(const LineScore l, const LineScore r) {
	double score1 = 0;
	double score2 = 0;
	if (l.totalPoints != 0) {
		score1 = double(l.goodPoints) / l.totalPoints;
	}
	if (r.totalPoints != 0) {
		score2 = double(r.goodPoints) / r.totalPoints;
	}
	return score1 > score2;
}

//Рандомное вещ. в интервале от a до b
float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

//Рандомное положительное целое в интервале от a до b
uint RandomInt(uint a, uint b) {
	uint output = a + (rand() % (uint)(b - a + 1));
	return output;
}

//Округление до целого по правилам округления
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

//Ищет номер линии по координатам точки
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

//Cost функция, которую минимизируем для расчета углов alpha и beta
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

//Минимизация cost function для двух пар пересекающихся линий
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

//Считает сколько пар линий стало ортогональными при найденных alpha и beta
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

//Алгоритм RANSAC (берет рандомные пары линий, делает минимизацию и считает сколько пар стало ортогональными, потом выбирает лучшее решение
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
		
		//Создали вектор как разность точек
		cv::Vec3f currentLine = endPoint - beginPoint;
		double currentLength = cv::norm(currentLine);
		cv::Vec3f normalizedCurrentLine;
		cv::normalize(currentLine, normalizedCurrentLine);
		
		int indexOfLeftIntersection = -1;
		int indexOfRightIntersection = -1;

		vector<cv::Point3f> leftPoints; //в этом векторе будут лежать точки, находящиеся в Octree в радиусе 10 пикселей от точки начала линии
		mainTree.getPointsWithinSphere(beginPoint, 10, leftPoints);
		for (int j = 0; j < leftPoints.size(); j++) 
		{
			cv::Point3f newBeginPoint = leftPoints[j];
			if (newBeginPoint != beginPoint) //Проверяем, что нашли точку другой линии
			{
				cv::Vec3f newLine = endPoint - newBeginPoint;
				double newLength = cv::norm(newLine);
				if (newLength != currentLength) //Если длины разные, то будем нормализовать
				{
					cv::Vec3f normalizedNewLine;
					cv::normalize(newLine, normalizedNewLine);
					cv::Vec3f difference = normalizedNewLine - normalizedCurrentLine;
					if (cv::norm(difference) <= threshold) //Если отклонение мало, то это та же линия, то есть будем продлевать
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

		vector<cv::Point3f> rightPoints; //в этом векторе будут лежать точки, находящиеся в Octree в радиусе 10 пикселей от точки конца линии
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
		
		//Добавляем продленную линию в наш вектор, с учетом порядка точек
		if (endPoint.x >= beginPoint.x) {
			ExtendedLinesVector.push_back(beginPoint);
			ExtendedLinesVector.push_back(endPoint);
		}
		else 
		{
			ExtendedLinesVector.push_back(endPoint);
			ExtendedLinesVector.push_back(beginPoint);
		}

		//Запоминаем с кем пересекается наша линия
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
	//Добавляем границы кадра как линии
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
		cv::Point3f middlePoint = cv::Point3f((endPoint.x - beginPoint.x) / 2, (endPoint.y - beginPoint.y) / 2, 0); //точка середины линии

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
	//Выделение групп параллельных линий
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
	
	//Дискритизация линий с шагом step
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
			PointIndexes.push_back(i); //запоминаем какая точка к какой линии принадлежит
			totalPoints++;
			currentX += step;
		}
		if (totalPoints != 0)
		{
			LineScoreVector[i].totalPoints = totalPoints;
		}
	}

	//Поиск линий в Octree
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
					
					//Смотрим какая линия левее
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
		
					//Поиск линии для точек начала
					vector<cv::Point3f> centersForSearchBegin;
					vector<LineScore> currentBeginScores = LineScoreVector;
					cv::Vec3f beginLineExpected = rightLineBeginPoint - leftLineBeginPoint;
					cv::Vec3f endLineExpected = rightLineEndPoint - leftLineEndPoint;
					double currentX = leftLineBeginPoint.x + step / 2;
					while (currentX <= rightLineBeginPoint.x) //дискритизация соединительных отрезков с шагом step/2
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
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a){ //идентифицируем к какой линии принадлежит найденная точка
								return a.x == tempPoint.x && a.y == tempPoint.y;
							});
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if (index != FirstIndex && index != SecondIndex)
							{
								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(beginLineExpected.dot(line) / (cv::norm(line)*cv::norm(beginLineExpected)));
								if (cosAngle > 0.95) { //фильтрация совсем трешовых вариантов
									currentBeginScores[index].goodPoints++;
								}
							}
						}
					}
					sort(currentBeginScores.begin(), currentBeginScores.end(), comparator); //сортируем
					uint BeginLineIndex = currentBeginScores[0].LineIndex; //выбираем лучшую

					//Поиск линии для точек начала
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
								
								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(endLineExpected.dot(line) / (cv::norm(line)*cv::norm(endLineExpected)));
								if (cosAngle > 0.95) {
									currentEndScores[index].goodPoints++;
								}
							}
						}
					}
					
					sort(currentEndScores.begin(), currentEndScores.end(), comparator);
					uint EndLineIndex = currentEndScores[0].LineIndex;

					if (currentEndScores[0].goodPoints != 0 && currentBeginScores[0].goodPoints != 0) { //если нашли обе линии, то сохраняем замкнутую область (по 4-ем индексам линий)
						vector<uint> FoundedPolygon = { FirstIndex, SecondIndex, BeginLineIndex, EndLineIndex };
						PolygonsVector.push_back(FoundedPolygon);
					}

				}
			}
		}
	}
	cout << "Found number of polygons: " << PolygonsVector.size() << endl;
}

//Надо переделать
void getNumbersOfPixels(cv::Mat image, vector<vector<uint>> PolygonsVector, int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<int>& NumberOfPixelsInArea) {
	for (int i = 0; i < PolygonsVector.size(); i++) {
		int counter = 0;

		cv::Mat copyImage = image.clone();
		vector<uint> indexVector = PolygonsVector[i];
		uint index1 = indexVector[0];
		uint index2 = indexVector[1];
		uint index3 = indexVector[2];
		uint index4 = indexVector[3];

		cv::Point3f B_1 = ExtendedLinesVector[index1 * 2];
		cv::Point3f E_1 = ExtendedLinesVector[index1 * 2 + 1];
		cv::Point3f B_2 = ExtendedLinesVector[index2 * 2];
		cv::Point3f E_2 = ExtendedLinesVector[index2 * 2 + 1];
		cv::Point3f B_3 = ExtendedLinesVector[index3 * 2];
		cv::Point3f E_3 = ExtendedLinesVector[index3 * 2 + 1];
		cv::Point3f B_4 = ExtendedLinesVector[index4 * 2];
		cv::Point3f E_4 = ExtendedLinesVector[index4 * 2 + 1];

		vector<cv::Point> tempVec = { cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), cv::Point(B_2.x, B_2.y), cv::Point(E_2.x, E_2.y), cv::Point(B_3.x, B_3.y), cv::Point(E_3.x, E_3.y), cv::Point(B_4.x, B_4.y), cv::Point(E_4.x, E_4.y) };

		const cv::Scalar blaColor = cv::Scalar(74, 184, 72, 255);
		cv::fillConvexPoly(copyImage, &tempVec[0], 4, blaColor);

		for (int x = 0; x < copyImage.cols; x++) {
			for (int y = 0; y < copyImage.rows; y++) {
				cv::Vec3b color = copyImage.at<cv::Vec3b>(cv::Point(x, y));
				if (color[0] == 74 && color[1] == 184 && color[2] == 72) {
					counter++;
				}
			}
		}
		NumberOfPixelsInArea.push_back(counter);
	}
}

void getIntersections(vector<vector<uint>> PolygonsVector, bool** &PolygonIntersections) {
	for (int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> lines = PolygonsVector[i];
		for (int j = 0; j < lines.size(); j++) {
			uint index = lines[j];
			for (int kek = i + 1; kek < PolygonsVector.size(); kek++) {
				vector<uint> currentLines = PolygonsVector[kek];
				if (currentLines[0] == index || currentLines[1] == index || currentLines[2] == index || currentLines[3] == index) {
					PolygonIntersections[i][kek] = 1;
					PolygonIntersections[kek][i] = 1;
				}
			}
		}
	}
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

//Wocc доделать
double DirectionScore(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines) {
	double result = 0;
	double Wocc = 1; // доделать подсчет этого коэффициента
	for (int i = 0; i < lines.size(); i++) {
		uint index = lines[i];
		if (direction == DirectionsOfLines[index]) {
			result += Wocc*lineSupportingScore(lines, index, numLinesDetected, LsdLinesVector, ExtendedLinesVector);
		}
	}
	return result;
}

double dataCost(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, int Warea) {
	double result = 0;
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

double smoothnessCost(vector<uint> linesA, vector<uint> linesB, uint directionA, uint directionB, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines) {
	int Csmooth = 5;
	if (directionA != directionB) {
		uint commonLine;
		for (int i = 0; i < linesA.size(); i++) {
			for (int j = 0; j < linesB.size(); j++) {
				if (linesA[i] = linesB[j]) {
					commonLine = linesA[i];
					break;
				}
			}
			break;
		}
		cv::Point3f lsdLineBegin = LsdLinesVector[2 * commonLine];
		cv::Point3f lsdLineEnd = LsdLinesVector[2 * commonLine + 1];
		double LSD_length = cv::norm(lsdLineEnd - lsdLineBegin);
		cv::Point3f commonLineBegin = ExtendedLinesVector[2 * commonLine];
		cv::Point3f commonLineEnd = ExtendedLinesVector[2 * commonLine + 1];
		double CommonLine_length = cv::norm(commonLineEnd - commonLineBegin);
		double Wl = 1 - 0.9*(LSD_length/CommonLine_length);
		double Wdir;
		if (directionA != DirectionsOfLines[commonLine] && directionB != DirectionsOfLines[commonLine]) {
			Wdir = 0.1;
		}
		else {
			Wdir = 1.0;
		}
		return Wdir*Wl*Csmooth;
	}
	return 0;
	
}

double EnergyFunction(vector<uint> PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint>> PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	double dataCostValue = 0;
	double smoothnessCostValue = 0;
	for (int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> lines = PolygonsVector[i];
		uint normalDirection = PlaneNormals[i];
		uint Warea = NumberOfPixelsInArea[i];
		dataCostValue += dataCost(lines, normalDirection, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, Warea);
		for (int j = i; j < PolygonsVector.size(); j++) {
			if (i != j && PolygonIntersections[i][j] == 1) {
				smoothnessCostValue += smoothnessCost(lines, PolygonsVector[j], normalDirection, PlaneNormals[j], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines);
			}
		}
	}
	return dataCostValue + smoothnessCostValue;
}

void mincut(int** g, int n, vector<int>& best_cut) {
	const int MAXN = 5000;
	int best_cost = 1000000000;
	vector<int> v[MAXN];
	for (int i = 0; i<n; ++i)
		v[i].assign(1, i);
	int w[MAXN];
	bool exist[MAXN], in_a[MAXN];
	memset(exist, true, sizeof exist);
	for (int ph = 0; ph<n - 1; ++ph) {
		memset(in_a, false, sizeof in_a);
		memset(w, 0, sizeof w);
		for (int it = 0, prev; it<n - ph; ++it) {
			int sel = -1;
			for (int i = 0; i<n; ++i)
			if (exist[i] && !in_a[i] && (sel == -1 || w[i] > w[sel]))
				sel = i;
			if (it == n - ph - 1) {
				if (w[sel] < best_cost)
					best_cost = w[sel], best_cut = v[sel];
				v[prev].insert(v[prev].end(), v[sel].begin(), v[sel].end());
				for (int i = 0; i<n; ++i)
					g[prev][i] = g[i][prev] += g[sel][i];
				exist[sel] = false;
			}
			else {
				in_a[sel] = true;
				for (int i = 0; i < n; ++i)
					w[i] += g[sel][i];
				prev = sel;
			}
		}
	}
}

void getNormals(vector<uint>& PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint>> PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	for (int i = 0; i < PolygonsVector.size(); i++) {
		uint randomNormalDirection = RandomInt(0,2);
		PlaneNormals.push_back(randomNormalDirection);
	}
	double InitialEnergyValue = EnergyFunction(PlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);
	
	bool success = 1;
	while (success)
	{
		success = 0;
		for (int swap = 0; swap < 3; swap++) //swap = 0, то меняем X и Y; swap = 1, то меняем Y и Z; swap = 2, то меняем X и Z
		{	
			uint AlphaDir, BetaDir;
			if (swap == 0) {
				AlphaDir = 0;
				BetaDir = 1;
			}
			else if (swap == 1) {
				AlphaDir = 1;
				BetaDir = 2;
			}
			else {
				AlphaDir = 0;
				BetaDir = 2;
			}

			//Строим граф
			int DIM = PolygonsVector.size() + 2;
			int **g = new int *[DIM];;
			for (int j = 0; j < DIM; j++) {   
				g[j] = new int[DIM];
			}
			g[DIM - 1][DIM - 1] = 0; g[DIM - 2][DIM - 2] = 0; g[DIM - 1][DIM - 2] = 0; g[DIM - 2][DIM - 1] = 0;

			for (int i = 0; i < DIM-2; i++) {
				if (PlaneNormals[i] != AlphaDir && PlaneNormals[i] != BetaDir) {
					int* zeroArray = new int[DIM];
					memset(zeroArray, 0, sizeof(zeroArray));
					g[i] = zeroArray;
					for (int k = 0; k < DIM; k++) {
						g[k][i] = 0;
					}
				}
				else {
					int dataCostAlpha = dataCost(PolygonsVector[i], AlphaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);
					int dataCostBeta = dataCost(PolygonsVector[i], BetaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);;
					double smoothnessCostAlpha = 0;
					for (int it = 0; it < PolygonsVector.size(); it++) {
						if (PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) {
							smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], AlphaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines);
						}
					}
					double smoothnessCostBeta = 0;
					for (int it = 0; it < PolygonsVector.size(); it++) {
						if (PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) {
							smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], BetaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines);
						}
					}
					g[i][DIM - 2] = dataCostAlpha + (int)RoundTo(smoothnessCostAlpha); //tlinkA
					g[i][DIM - 1] = dataCostBeta + (int)RoundTo(smoothnessCostBeta); // tlinkB
					for (int j = i; j < DIM - 2; j++) {
						
						if (i == j || PolygonIntersections[i][j] == 0) {
							g[i][j] = 0;
						}
						else {
							if (PlaneNormals[j] == AlphaDir || PlaneNormals[j] == BetaDir) {
								g[i][j] = (int)RoundTo(smoothnessCost(PolygonsVector[i], PolygonsVector[j], AlphaDir, BetaDir, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines));
							}
							else {
								g[i][j] = 0;
							}
						}
					}
				}
			}
			for (int i = 0; i < DIM; i++) {
				for (int j = 0; j < i; j++) {
					g[i][j] = g[j][i];
				}
			}
			
			//делаем минимальный разрез
			vector<int> minimumCut;
			vector<uint> newPlaneNormals = PlaneNormals;
			mincut(g, DIM, minimumCut);
			auto pos = find(minimumCut.begin(), minimumCut.end(), DIM-2);
			if (pos != minimumCut.end()) {
				for (int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if (position != minimumCut.end()) {
						newPlaneNormals[kek] = BetaDir;
					}
					else {
						newPlaneNormals[kek] = AlphaDir;
					}
				}
			}
			else {
				for (int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if (position != minimumCut.end()) {
						newPlaneNormals[kek] = AlphaDir;
					}
					else {
						newPlaneNormals[kek] = BetaDir;
					}
				}
			}

			double EnergyValue_With_Hat = EnergyFunction(newPlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);
			if (EnergyValue_With_Hat < InitialEnergyValue)
			{
				PlaneNormals = newPlaneNormals;
				InitialEnergyValue = EnergyValue_With_Hat;
				success = 1;
			}
		}	
	}
}








int main()
{	
	clock_t tStart = clock();
	srand(time(0)); //чтобы последовательность рандомных чисел была всегда уникальна при новом запуске
	
	//Переменные для настройки
	double focal_length = 4; //фокальное расстояние в mm
	double sensor_width = 4.59; //ширина сенсора в mm
	double ExtendThreshold = 0.01; //порог отклонения линии (для удлинения)
	double countInlierThreshold = 0.0001; //если квадрат скалярного произведения двух линий меньше этого числа, то мы считаем эти линии ортогональными
	
	double AngleTolerance = 0.0001; //если abs(tg(angle1) - tg(angle2)) < AngleTolerance, то эти две линии объединяются в одну группу (параллельных линий с некоторой степенью толерантности)
	double step = 6; //шаг для дискритизации линий
	double radius = 20; //радиус поиска около точек соединительных линий
	
	uint ResizeIfMoreThan = 2000; //если ширина или высота изображения больше этого числа, то мы меняем размер изображения
	bool debug = 1;
	
	//Открытие изображения
	cv::Mat image;
	image = cv::imread("test.jpg", CV_LOAD_IMAGE_COLOR);
	
	//Изменение размера
	uint maxRes = max(image.cols, image.rows);
	if (maxRes > ResizeIfMoreThan)
	{
		float scaleFactor = float(ResizeIfMoreThan) / maxRes;
		cv::Size size = cv::Size(image.cols*scaleFactor, image.rows*scaleFactor);
		cv::resize(image, image, size);
	}
	
	//Границы кадра
	uint maxX = image.cols;
	uint maxY = image.rows;
	
	//Фокальное расстояние
	double temp_focal = maxX*focal_length / sensor_width;
	uint f = (uint)temp_focal; //фокальное расстояние в пикселях

	//LSD
	int numLinesDetected;
	double* LsdLinesArray = DoLSD(image, numLinesDetected);
	cout << "Number of LSD lines detected: " << numLinesDetected << endl;

	//Копируем точки из массива в вектор с учетом порядка
	vector<cv::Point3f> LsdLinesVector; //элемент с номером 2*i дает нам точку начала i-ой линии, элемент с номером 2*i + 1 дает нам точку конца i-ой линии
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

	//Удлинение
	vector<cv::Point3f> ExtendedLinesVector; //элемент с индексом 2*i дает нам точку начала i-ой линии, элемент с индексом 2*i + 1 дает нам точку конца i-ой линии
	vector<PairOfTwoLines> LinePairsVector; //в каждом элементе этого вектора лежит два индекса (линий, которые пересекаются)
	ExtendLines(numLinesDetected, LsdLinesVector, ExtendedLinesVector, LinePairsVector, ExtendThreshold, maxX, maxY);
	cout << "Number of line pairs: " << LinePairsVector.size() << endl;
	
	//RANSAC (находим углы альфа и бэта)
	uint maxRansacTrials = (uint)LinePairsVector.size(); //максимальное количество итерация алгоритма RANSAC
	dlib::matrix<double, 0, 1> solution = RANSAC(maxRansacTrials, countInlierThreshold, f, LinePairsVector, ExtendedLinesVector);
	cout << "angle Alpha (in radians): " << solution(0) << endl << "angle Beta (in radians): " << solution(1) << endl;

	//Получаем vanishing points
	vector<cv::Point3f> VanishingPoints;
	getVanishingPoints(solution(0), solution(1), f, VanishingPoints);

	//Находим направление каждой линии
	vector<uint> DirectionsOfLines; //элемент с номером i означает направление i-ой линии (0=x, 1=y, 2=z, 3=xy, 4=xz, 5=yz)
	assignDirections(numLinesDetected, ExtendedLinesVector, VanishingPoints, DirectionsOfLines);

	//Выделяем все замкнутые области
	vector<vector<uint>> PolygonsVector;
	getPolygons(numLinesDetected, ExtendedLinesVector, AngleTolerance, step, radius, PolygonsVector);

	
	//Смотрим сколько пикселей в каждой области
	vector<int> NumbersOfPixelsInArea;
	getNumbersOfPixels(image, PolygonsVector, numLinesDetected, ExtendedLinesVector, NumbersOfPixelsInArea);
	
	//Запоминаем какие области с кем являются соседними
	int sizeOfPolInter = PolygonsVector.size();
	bool **PolygonIntersections = new bool *[sizeOfPolInter];;
	for (int j = 0; j < sizeOfPolInter; j++) {
		PolygonIntersections[j] = new bool[sizeOfPolInter];
	}
	for (int k1 = 0; k1 < sizeOfPolInter; k1++) {
		for (int k2 = 0; k2 < sizeOfPolInter; k2++) {
			PolygonIntersections[k1][k2] = 0;
		}
	}
	getIntersections(PolygonsVector, PolygonIntersections);

	//Получаем нормали
	vector<uint> PlaneNormals;
	getNormals(PlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumbersOfPixelsInArea, PolygonIntersections);









	//Запись в файл для дебага
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
		
		cv::Mat copyImage2 = cv::imread("test.jpg");
		for (int i = 0; i < numLinesDetected; i++) {
			
			if (DirectionsOfLines[i] == 0) {
				const cv::Scalar blaColor = cv::Scalar(255, 0, 0, 255);
				cv::Point3f B_1 = ExtendedLinesVector[i * 2];
				cv::Point3f E_1 = ExtendedLinesVector[i * 2 + 1];
				cv::line(copyImage2, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 3);
			}
			if (DirectionsOfLines[i] == 1) {
				const cv::Scalar blaColor = cv::Scalar(0, 255, 0, 255);
				cv::Point3f B_1 = ExtendedLinesVector[i * 2];
				cv::Point3f E_1 = ExtendedLinesVector[i * 2 + 1];
				cv::line(copyImage2, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 3);
			}
			if (DirectionsOfLines[i] == 2) {
				const cv::Scalar blaColor = cv::Scalar(0, 0, 255, 255);
				cv::Point3f B_1 = ExtendedLinesVector[i * 2];
				cv::Point3f E_1 = ExtendedLinesVector[i * 2 + 1];
				cv::line(copyImage2, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 3);
			}
		
		}
		cv::imwrite("directions.jpg", copyImage2);

		
		cv::Mat copyImage = image.clone();
		for (int i = 0; i < PolygonsVector.size(); i++) {
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
			vector<cv::Point> tempVec = { cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), cv::Point(E_4.x, E_4.y), cv::Point(E_3.x, E_3.y) };
			const cv::Scalar blaColor = cv::Scalar(RandomInt(0, 255), RandomInt(0, 255), RandomInt(0, 255), 255);
			cv::fillConvexPoly(copyImage, &tempVec[0], 4, blaColor);
			//cv::line(copyImage, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_2.x, B_2.y), cv::Point(E_2.x, E_2.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_3.x, B_3.y), cv::Point(E_3.x, E_3.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_4.x, B_4.y), cv::Point(E_4.x, E_4.y), blaColor, 5);
		}
		cv::imwrite("all_polygons.jpg", copyImage);

		cv::Mat secondImage = image.clone();
		for (int i = 0; i < PolygonsVector.size(); i++) {
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
			if (PlaneNormals[i] == 0) {
				const cv::Scalar blaColor = cv::Scalar(0, 0, 255, 255);
				cv::fillConvexPoly(secondImage, &tempVec[0], 4, blaColor);
			}
			else if (PlaneNormals[i] == 1) {
				const cv::Scalar blaColor = cv::Scalar(0, 255, 0, 255);
				cv::fillConvexPoly(secondImage, &tempVec[0], 4, blaColor);
			}
			else if (PlaneNormals[i] == 2) {
				const cv::Scalar blaColor = cv::Scalar(255, 0, 0, 255);
				cv::fillConvexPoly(secondImage, &tempVec[0], 4, blaColor);
			}
			//cv::line(copyImage, cv::Point(B_1.x, B_1.y), cv::Point(E_1.x, E_1.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_2.x, B_2.y), cv::Point(E_2.x, E_2.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_3.x, B_3.y), cv::Point(E_3.x, E_3.y), blaColor, 5);
			//cv::line(copyImage, cv::Point(B_4.x, B_4.y), cv::Point(E_4.x, E_4.y), blaColor, 5);
		}
		cv::imwrite("normals.jpg", secondImage);
	}
	system("pause");
	return 0;
}

