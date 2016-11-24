//C++ Includes
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <chrono>
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
#include "lsd/lsd.h"
#include "boolinq/boolinq.h"
using namespace std;

//Записывает LSD линии в файл .eps
void write_eps(double* segs, int n, int dim, char* filename, int xsize, int ysize, double width) {
	FILE* eps;
	int i;

	/* open file */
	if(strcmp(filename, "-") == 0)
		eps = stdout;
	else
		eps = fopen(filename, "w");

	/* write EPS header */
	fprintf(eps, "%%!PS-Adobe-3.0 EPSF-3.0\n");
	fprintf(eps, "%%%%BoundingBox: 0 0 %d %d\n", xsize, ysize);
	fprintf(eps, "%%%%Creator: LSD, Line Segment Detector\n");
	fprintf(eps, "%%%%Title: (%s)\n", filename);
	fprintf(eps, "%%%%EndComments\n");

	/* write line segments */
	for(i = 0; i < n; i++) {
		fprintf(eps, "newpath %f %f moveto %f %f lineto %f setlinewidth stroke\n",
				segs[i * dim + 0],
				(double)ysize - segs[i * dim + 1],
				segs[i * dim + 2],
				(double)ysize - segs[i * dim + 3],
				width <= 0.0 ? segs[i * dim + 4] : width);
	}

	/* close EPS file */
	fprintf(eps, "showpage\n");
	fprintf(eps, "%%%%EOF\n");
}

//LSD
double* DoLSD(cv::Mat image, int& numLines) {
	//Конвертация изображения для последующего применения алгоритма LSD
	cv::Mat grayscaleMat(image.size(), CV_8U);
	cv::cvtColor(image, grayscaleMat, CV_BGR2GRAY);
	double* pgm_image;
	pgm_image = (double *)malloc(image.cols * image.rows * sizeof(double));
	for(int y = 0; y < image.rows; y++) {
		for(int x = 0; x < image.cols; x++) {
			int i = x + (y * image.cols);
			pgm_image[i] = double(grayscaleMat.data[i]);
		}
	}
	//Применение LSD
	int numLinesDetected;
	double* outArray;
	outArray = lsd(&numLinesDetected, pgm_image, image.cols, image.rows);
	numLines = numLinesDetected;
	//write_eps(outArray, numLinesDetected, 7, "lsd.eps", image.cols, image.rows, 0);
	return outArray;
}

//Округление до целого по правилам округления
double RoundTo(double x) {
	int y = floor(x);
	if((x - y) >= 0.5)
		y++;
	return (double)y;
}

class Line {
public:
	cv::Point3f begin;
	cv::Point3f end;
	int index;

	Line(cv::Point3f a = cv::Point3f(-1, -1, -1), cv::Point3f b = cv::Point3f(-1, -1, -1), int index = -1):
		begin(a), end(b), index(index) {};
	double norm() {
		double norm = sqrt((end.x - begin.x)*(end.x - begin.x) + (end.y - begin.y)*(end.y - begin.y));
		return RoundTo(norm);
	};
};

class PairOfTwoLines {
public:
	uint FirstIndex;
	uint SecondIndex;

	PairOfTwoLines(uint a, uint b) {
		FirstIndex = a;
		SecondIndex = b;
	}

	PairOfTwoLines() {
		FirstIndex = 0;
		SecondIndex = 0;
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
	if(l.totalPoints != 0) { score1 = double(l.goodPoints) / l.totalPoints; }
	if(r.totalPoints != 0) { score2 = double(r.goodPoints) / r.totalPoints; }
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

//Intrinsic matrix
cv::Mat K_inv(double f) {
	double temp[3][3] = {{f, 0, 0}, {0, f, 0}, {0, 0, 1}};
	cv::Mat k = cv::Mat(3, 3, CV_64F, temp);
	return k.inv();
}

//Ищет номер линии по координатам точки
int FindIndexOfLine(vector<cv::Point3f> vec, cv::Point3f point) {
	auto position = find_if(vec.begin(), vec.end(), [&](const cv::Point3f& a) { return a.x == point.x && a.y == point.y; });
	if(position != vec.end()) {
		int index = (position - vec.begin()) / 2;
		return index;
	}
	return -1;
}

//Cost функция, которую минимизируем для расчета углов alpha и beta
class cost_function {
private:

public:
	cv::Mat lineOne;
	cv::Mat lineTwo;
	cv::Mat lineThree;
	cv::Mat lineFour;
	cv::Mat K_inverted;

	cost_function(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h) {
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

	double operator()(const dlib::matrix<double, 0, 1>& arg) const {
		double summ = 0;

		double r_x_array[3][3] = {{1, 0, 0}, {0, cos(arg(0)), -sin(arg(0))}, {0, sin(arg(0)), cos(arg(0))}};
		cv::Mat R_x = cv::Mat(3, 3, CV_64F, r_x_array);
		double r_y_array[3][3] = {{cos(arg(1)), 0, sin(arg(1))}, {0, 1, 0}, {-sin(arg(1)), 0, cos(arg(1))}};
		cv::Mat R_y = cv::Mat(3, 3, CV_64F, r_y_array);
		cv::Mat H = R_y * R_x * K_inverted;
		cv::Mat H_t = H.t();

		cv::Mat l_1 = H_t * lineOne;
		cv::normalize(l_1, l_1);
		l_1.resize(2);

		cv::Mat l_2 = H_t * lineTwo;
		cv::normalize(l_2, l_2);
		l_2.resize(2);

		cv::Mat l_3 = H_t * lineThree;
		cv::normalize(l_3, l_3);
		l_3.resize(2);

		cv::Mat l_4 = H_t * lineFour;
		cv::normalize(l_4, l_4);
		l_4.resize(2);

		cv::Mat multiply_one_mat = l_1.t() * l_2;
		cv::Mat multiply_two_mat = l_3.t() * l_4;
		summ += multiply_one_mat.at<double>(0, 0) * multiply_one_mat.at<double>(0, 0) + multiply_two_mat.at<double>(0, 0) * multiply_two_mat.at<double>(0, 0);
		return summ;
	}
};

//Минимизация cost function для двух пар пересекающихся линий
dlib::matrix<double, 0, 1> minimize_C(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h) {
	dlib::matrix<double, 0, 1> solution(2);
	solution = 0, 0;
	find_min_bobyqa(cost_function(f, a, b, c, d, e, ff, g, h),
					solution,
					5, // number of interpolation points
					dlib::uniform_matrix<double>(2, 1, -M_PI / 2), // lower bound constraint
					dlib::uniform_matrix<double>(2, 1, M_PI / 2), // upper bound constraint
					M_PI / 10, // initial trust region radius
					0.001, // stopping trust region radius
					100 // max number of objective function evaluations
	);
	return solution;
}

//Считает сколько пар линий стало ортогональными при найденных alpha и beta
uint countInlierScore(dlib::matrix<double, 0, 1> solution, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint score = 0;
	cv::Mat K_inverted = K_inv(f);
	double temp[3][3] = {{1, 0, 0}, {0, cos(solution(0)), -sin(solution(0))}, {0, sin(solution(0)), cos(solution(0))}};
	cv::Mat R_x = cv::Mat(3, 3, CV_64F, temp);
	double temp2[3][3] = {{cos(solution(1)), 0, sin(solution(1))}, {0, 1, 0}, {-sin(solution(1)), 0, cos(solution(1))}};
	cv::Mat R_y = cv::Mat(3, 3, CV_64F, temp2);
	cv::Mat H = R_y * R_x * K_inverted;
	cv::Mat H_t = H.t();
	for(int i = 0; i < LinePairsVector.size(); i++) {
		PairOfTwoLines pair = LinePairsVector[i];
		cv::Vec3f lineOne = ExtendedLinesVector[2 * pair.FirstIndex + 1] - ExtendedLinesVector[2 * pair.FirstIndex];
		cv::Vec3f lineTwo = ExtendedLinesVector[2 * pair.SecondIndex + 1] - ExtendedLinesVector[2 * pair.SecondIndex];

		cv::Mat first_line = cv::Mat(lineOne);
		first_line.convertTo(first_line, CV_64F);
		cv::Mat second_line = cv::Mat(lineTwo);
		second_line.convertTo(second_line, CV_64F);

		cv::Mat l_1 = H_t * first_line;
		cv::normalize(l_1, l_1);
		l_1.resize(2);

		cv::Mat l_2 = H_t * second_line;
		cv::normalize(l_2, l_2);
		l_2.resize(2);

		cv::Mat multiply_one_mat = l_1.t() * l_2;
		double summ = multiply_one_mat.at<double>(0, 0) * multiply_one_mat.at<double>(0, 0);
		if(summ <= threshold) { score++; }
	}
	return score;
}

//Алгоритм RANSAC (берет рандомные пары линий, делает минимизацию и считает сколько пар стало ортогональными, потом выбирает лучшее решение
dlib::matrix<double, 0, 1> RANSAC(uint maxTrials, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint counter = 0;
	uint bestScore = 0;
	dlib::matrix<double, 0, 1> bestSolution(2);
	while(counter < maxTrials) {
		uint first_index = RandomInt(0, LinePairsVector.size() - 1);
		uint second_index = first_index;
		while(second_index == first_index) { second_index = RandomInt(0, LinePairsVector.size() - 1); }
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
		if(score > bestScore) {
			bestScore = score;
			bestSolution = solution;
		}
		counter++;
	}
	cout << "Number of inliers: " << bestScore << endl;
	return bestSolution;
}

//http://stackoverflow.com/questions/16792751/hashmap-for-2d3d-coordinates-i-e-vector-of-doubles
struct hashFunc {
	size_t operator()(const cv::Point3f& k) const {
		auto d = *(double*)(&k.x) + k.z;
		return hash<double>()(d);
		//size_t h1 =ha sh<float>()(k.x);
		//size_t h2 = hash<float>()(k.y);
		//size_t h3 = hash<float>()(k.z);
		//return (h1 ^ (h2 << 1)) ^ h3;
	}
};

struct equalsFunc {
	bool operator()(const cv::Point3f& lhs, const cv::Point3f& rhs) const { return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z); }
};

void getVanishingPoints(double alpha, double beta, double f, vector<cv::Point3f>& output) {

	double kArray[3][3] = {{f, 0, 0}, {0, f, 0}, {0, 0, 1}};
	cv::Mat K_matrix = cv::Mat(3, 3, CV_64F, kArray);
	double r_x_array[3][3] = {{1, 0, 0}, {0, cos(alpha), -sin(alpha)}, {0, sin(alpha), cos(alpha)}};
	cv::Mat Rx = cv::Mat(3, 3, CV_64F, r_x_array);
	double r_y_array[3][3] = {{cos(beta), 0, sin(beta)}, {0, 1, 0}, {-sin(beta), 0, cos(beta)}};
	cv::Mat Ry = cv::Mat(3, 3, CV_64F, r_y_array);

	cv::Mat R = Ry * Rx;
	cv::Mat R_t = R.t();

	double XVectorArray[3][1] = {{0}, {0}, {1}};
	cv::Mat XMat = cv::Mat(3, 1, CV_64F, XVectorArray);
	double YVectorArray[3][1] = {{0}, {1}, {0}};
	cv::Mat YMat = cv::Mat(3, 1, CV_64F, YVectorArray);
	double ZVectorArray[3][1] = {{1}, {0}, {0}};
	cv::Mat ZMat = cv::Mat(3, 1, CV_64F, ZVectorArray);

	cv::Mat Nx = R_t * XMat;
	cv::Mat Ny = R_t * YMat;
	cv::Mat Nz = R_t * ZMat;

	cv::Mat Vx_m = K_matrix * Nx;
	cv::Mat Vy_m = K_matrix * Ny;
	cv::Mat Vz_m = K_matrix * Nz;

	cv::Point3f Vx = cv::Point3f(Vx_m);
	cv::Point3f Vy = cv::Point3f(Vy_m);
	cv::Point3f Vz = cv::Point3f(Vz_m);

	output.push_back(Vx);
	output.push_back(Vy);
	output.push_back(Vz);
}

void assignDirections(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<cv::Point3f> VanishingPoints, vector<uint>& output) {
	for(int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		cv::Vec3f line = endPoint - beginPoint;
		cv::Point3f middlePoint = cv::Point3f((endPoint.x - beginPoint.x) / 2, (endPoint.y - beginPoint.y) / 2, 0); //точка середины линии

		cv::Vec3f lineToX = middlePoint - VanishingPoints[0];
		double cosX = abs(line.dot(lineToX) / (cv::norm(line) * cv::norm(lineToX)));
		cv::Vec3f lineToY = middlePoint - VanishingPoints[1];
		double cosY = abs(line.dot(lineToY) / (cv::norm(line) * cv::norm(lineToY)));
		cv::Vec3f lineToZ = middlePoint - VanishingPoints[2];
		double cosZ = abs(line.dot(lineToZ) / (cv::norm(line) * cv::norm(lineToZ)));

		uint result;
		if(cosX <= 0.9 && cosY <= 0.9 && abs(cosX - cosY) < 0.01) { result = 3; } else if(cosX <= 0.9 && cosZ <= 0.9 && abs(cosX - cosZ) < 0.01) { result = 4; } else if(cosY <= 0.9 && cosZ <= 0.9 && abs(cosY - cosZ) < 0.01) { result = 5; } else if(cosX <= 0.9) { result = 0; } else if(cosY <= 0.9) { result = 1; } else
			if(cosZ <= 0.9) { result = 2; }
		output.push_back(result);
	}
}

void getPolygons(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, double AngleTolerance, double step, double radius, vector<vector<uint> >& PolygonsVector) {
	//Выделение групп параллельных линий
	vector<vector<uint> > ParallelLineGroups;
	vector<uint> FirstGroup;
	ParallelLineGroups.push_back(FirstGroup);
	vector<double> GroupCoefficients;
	GroupCoefficients.push_back(0);
	vector<uint> VerticalLinesGroup;
	for(int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		if(endPoint.x - beginPoint.x != 0) {
			double AngleCoef = (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x);
			bool Found = false;
			for(int j = 0; j < GroupCoefficients.size(); j++) {
				if(abs(AngleCoef - GroupCoefficients[j]) < AngleTolerance) {
					ParallelLineGroups[j].push_back(i);
					Found = true;
				}
			}
			if(!Found) {
				GroupCoefficients.push_back(AngleCoef);
				vector<uint> Group;
				Group.push_back(i);
				ParallelLineGroups.push_back(Group);
			}
		} else { VerticalLinesGroup.push_back(i); }
	}
	ParallelLineGroups.push_back(VerticalLinesGroup);

	vector<LineScore> LineScoreVector;
	for(int i = 0; i < numLinesDetected; i++) {
		LineScore temp = LineScore(0, 0, i);
		LineScoreVector.push_back(temp);
	}

	//Дискритизация линий с шагом step
	vector<cv::Point3f> pointsForSearch;
	vector<uint> PointIndexes;
	for(int i = 0; i < numLinesDetected; i++) {
		uint totalPoints = 0;
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		double currentX = beginPoint.x + step;
		while(currentX <= endPoint.x) {
			double currentY = (currentX - beginPoint.x) * (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x) + beginPoint.y;
			cv::Point3f point = cv::Point3f(currentX, currentY, 0);
			pointsForSearch.push_back(point);
			PointIndexes.push_back(i); //запоминаем какая точка к какой линии принадлежит
			totalPoints++;
			currentX += step;
		}
		if(totalPoints != 0) { LineScoreVector[i].totalPoints = totalPoints; }
	}

	//Поиск линий в Octree
	cv::Octree tree = cv::Octree(pointsForSearch);
	for(int i = 0; i < ParallelLineGroups.size(); i++) {
		vector<uint> Group = ParallelLineGroups[i];
		if(Group.size() >= 2) {
			for(int j = 0; j < Group.size(); j++) {

				for(int k = j + 1; k < Group.size(); k++) {

					uint FirstIndex = Group[j];
					uint SecondIndex = Group[k];
					cv::Point3f leftLineBeginPoint;
					cv::Point3f leftLineEndPoint;
					cv::Point3f rightLineBeginPoint;
					cv::Point3f rightLineEndPoint;

					//Смотрим какая линия левее
					if(ExtendedLinesVector[2 * SecondIndex + 1].x >= ExtendedLinesVector[2 * FirstIndex + 1].x) {
						leftLineBeginPoint = ExtendedLinesVector[2 * FirstIndex];
						leftLineEndPoint = ExtendedLinesVector[2 * FirstIndex + 1];
						rightLineBeginPoint = ExtendedLinesVector[2 * SecondIndex];
						rightLineEndPoint = ExtendedLinesVector[2 * SecondIndex + 1];
					} else {
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
					while(currentX <= rightLineBeginPoint.x) //дискритизация соединительных отрезков с шагом step/2
					{
						double currentY = (currentX - leftLineBeginPoint.x) * (rightLineBeginPoint.y - leftLineBeginPoint.y) / (rightLineBeginPoint.x - leftLineBeginPoint.x) + leftLineBeginPoint.y;
						cv::Point3f point = cv::Point3f(currentX, currentY, 0);
						centersForSearchBegin.push_back(point);
						currentX += step / 2;
					}
					for(int g = 0; g < centersForSearchBegin.size(); g++) {
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchBegin[g], radius, foundedPoints);
						for(int h = 0; h < foundedPoints.size(); h++) {
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a) { //идентифицируем к какой линии принадлежит найденная точка
								return a.x == tempPoint.x && a.y == tempPoint.y;
							});
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if(index != FirstIndex && index != SecondIndex) {
								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(beginLineExpected.dot(line) / (cv::norm(line) * cv::norm(beginLineExpected)));
								if(cosAngle > 0.95) { //фильтрация совсем трешовых вариантов
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
					while(currentX <= rightLineEndPoint.x) {
						double currentY = (currentX - leftLineEndPoint.x) * (rightLineEndPoint.y - leftLineEndPoint.y) / (rightLineEndPoint.x - leftLineEndPoint.x) + leftLineEndPoint.y;
						cv::Point3f point = cv::Point3f(currentX, currentY, 0);
						centersForSearchEnd.push_back(point);
						currentX += step / 2;
					}
					for(int g = 0; g < centersForSearchEnd.size(); g++) {
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchEnd[g], radius, foundedPoints);
						for(int h = 0; h < foundedPoints.size(); h++) {
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a) { return a.x == tempPoint.x && a.y == tempPoint.y; });
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if(index != FirstIndex && index != SecondIndex) {

								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(endLineExpected.dot(line) / (cv::norm(line) * cv::norm(endLineExpected)));
								if(cosAngle > 0.95) { currentEndScores[index].goodPoints++; }
							}
						}
					}

					sort(currentEndScores.begin(), currentEndScores.end(), comparator);
					uint EndLineIndex = currentEndScores[0].LineIndex;

					if(currentEndScores[0].goodPoints != 0 && currentBeginScores[0].goodPoints != 0) { //если нашли обе линии, то сохраняем замкнутую область (по 4-ем индексам линий)
						vector<uint> FoundedPolygon = {FirstIndex, SecondIndex, BeginLineIndex, EndLineIndex};
						PolygonsVector.push_back(FoundedPolygon);
					}

				}
			}
		}
	}
	cout << "Found number of polygons: " << PolygonsVector.size() << endl;
}

void getNumbersOfPixels(cv::Mat image, vector<vector<uint> > PolygonsVector, int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<int>& NumberOfPixelsInArea) {
	for(int i = 0; i < PolygonsVector.size(); i++) {
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

		double side1 = cv::norm(E_1 - B_1);
		double side2 = cv::norm(E_2 - B_2);
		double side3 = cv::norm(E_3 - B_3);
		double side4 = cv::norm(E_4 - B_4);

		double p = (side1 + side2 + side3 + side4) / 2;
		double result = sqrt((p - side1) * (p - side2) * (p - side3) * (p - side4));
		counter = RoundTo(result);
		NumberOfPixelsInArea.push_back(counter);
	}
}

void getIntersections(vector<vector<uint> > PolygonsVector, bool** & PolygonIntersections) {
	for(int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> linesA = PolygonsVector[i];

		for(int j = i+1; j < PolygonsVector.size(); j++) {
			vector<uint> linesB = PolygonsVector[j];
			
			for (auto &lineA : linesA) {
				for (auto &lineB : linesB) {
					if (lineA == lineB) {
						PolygonIntersections[i][j] = 1;
						PolygonIntersections[j][i] = 1;
						break;
					}
				}
				break;
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

	for(int i = 0; i < lines.size(); i++) {
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
	for(int i = 0; i < lines.size(); i++) {
		uint index = lines[i];
		if(direction == DirectionsOfLines[index]) { result += Wocc * lineSupportingScore(lines, index, numLinesDetected, LsdLinesVector, ExtendedLinesVector); }
	}
	return result;
}

double dataCost(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, double Warea) {
	double result = 0;
	if(direction == 0) { result = -Warea * (DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 5, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); } else if(direction == 1) { result = -Warea * (DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 4, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); } else
		if(direction == 2) { result = -Warea * (DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 3, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); }
	return result;

}

double smoothnessCost(vector<uint> linesA, vector<uint> linesB, uint directionA, uint directionB, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines) {
	int Csmooth = 5;
	if(directionA != directionB) {
		uint commonLine;
		for(int i = 0; i < linesA.size(); i++) {
			for(int j = 0; j < linesB.size(); j++) {
				if(linesA[i] = linesB[j]) {
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
		double Wl = 1 - 0.9 * (LSD_length / CommonLine_length);
		double Wdir;
		if(directionA != DirectionsOfLines[commonLine] && directionB != DirectionsOfLines[commonLine]) { Wdir = 0.1; } else { Wdir = 1.0; }
		return Wdir * Wl * Csmooth;
	}
	return 0;

}

double EnergyFunction(vector<uint> PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	double dataCostValue = 0;
	double smoothnessCostValue = 0;
	for(int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> lines = PolygonsVector[i];
		uint normalDirection = PlaneNormals[i];
		double Warea = 1 + (double)NumberOfPixelsInArea[i] / 100;
		dataCostValue += dataCost(lines, normalDirection, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, Warea);
		for(int j = i; j < PolygonsVector.size(); j++) { if(i != j && PolygonIntersections[i][j] == 1) { smoothnessCostValue += smoothnessCost(lines, PolygonsVector[j], normalDirection, PlaneNormals[j], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
	}
	return dataCostValue + smoothnessCostValue;
}

void mincut(int** g, int n, vector<int>& best_cut) {
	const int MAXN = 5000;
	int best_cost = 1000000000;
	vector<int> v[MAXN];
	for(int i = 0; i < n; ++i)
		v[i].assign(1, i);
	int w[MAXN];
	bool exist[MAXN], in_a[MAXN];
	memset(exist, true, sizeof exist);
	for(int ph = 0; ph < n - 1; ++ph) {
		memset(in_a, false, sizeof in_a);
		memset(w, 0, sizeof w);
		for(int it = 0, prev; it < n - ph; ++it) {
			int sel = -1;
			for(int i = 0; i < n; ++i)
				if(exist[i] && !in_a[i] && (sel == -1 || w[i] > w[sel]))
					sel = i;
			if(it == n - ph - 1) {
				if(w[sel] < best_cost)
					best_cost = w[sel], best_cut = v[sel];
				v[prev].insert(v[prev].end(), v[sel].begin(), v[sel].end());
				for(int i = 0; i < n; ++i)
					g[prev][i] = g[i][prev] += g[sel][i];
				exist[sel] = false;
			} else {
				in_a[sel] = true;
				for(int i = 0; i < n; ++i)
					w[i] += g[sel][i];
				prev = sel;
			}
		}
	}
}

void getNormals(vector<uint>& PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	for(int i = 0; i < PolygonsVector.size(); i++) {
		uint randomNormalDirection = RandomInt(0, 2);
		PlaneNormals.push_back(randomNormalDirection);
	}
	double InitialEnergyValue = EnergyFunction(PlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);

	bool success = 1;
	while(success) {
		success = 0;
		for(int swap = 0; swap < 3; swap++) //swap = 0, то меняем X и Y; swap = 1, то меняем Y и Z; swap = 2, то меняем X и Z
		{
			uint AlphaDir, BetaDir;
			if(swap == 0) {
				AlphaDir = 0;
				BetaDir = 1;
			} else if(swap == 1) {
				AlphaDir = 1;
				BetaDir = 2;
			} else {
				AlphaDir = 0;
				BetaDir = 2;
			}

			//Строим граф
			int DIM = PolygonsVector.size() + 2;
			int** g = new int *[DIM];;
			for(int j = 0; j < DIM; j++) { g[j] = new int[DIM]; }
			g[DIM - 1][DIM - 1] = 0;
			g[DIM - 2][DIM - 2] = 0;
			g[DIM - 1][DIM - 2] = 0;
			g[DIM - 2][DIM - 1] = 0;

			for(int i = 0; i < DIM - 2; i++) {
				if(PlaneNormals[i] != AlphaDir && PlaneNormals[i] != BetaDir) {
					int* zeroArray = new int[DIM];
					memset(zeroArray, 0, sizeof(zeroArray));
					g[i] = zeroArray;
					for(int k = 0; k < DIM; k++) { g[k][i] = 0; }
				} else {
					int dataCostAlpha = dataCost(PolygonsVector[i], AlphaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);
					int dataCostBeta = dataCost(PolygonsVector[i], BetaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);;
					double smoothnessCostAlpha = 0;
					for(int it = 0; it < PolygonsVector.size(); it++) { if(PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) { smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], AlphaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
					double smoothnessCostBeta = 0;
					for(int it = 0; it < PolygonsVector.size(); it++) { if(PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) { smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], BetaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
					g[i][DIM - 2] = dataCostAlpha + (int)RoundTo(smoothnessCostAlpha); //tlinkA
					g[i][DIM - 1] = dataCostBeta + (int)RoundTo(smoothnessCostBeta); // tlinkB
					for(int j = i; j < DIM - 2; j++) {

						if(i == j || PolygonIntersections[i][j] == 0) { g[i][j] = 0; } else {
							if(PlaneNormals[j] == AlphaDir || PlaneNormals[j] == BetaDir) { g[i][j] = (int)RoundTo(smoothnessCost(PolygonsVector[i], PolygonsVector[j], AlphaDir, BetaDir, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); } else { g[i][j] = 0; }
						}
					}
				}
			}
			for(int i = 0; i < DIM; i++) { for(int j = 0; j < i; j++) { g[i][j] = g[j][i]; } }

			//делаем минимальный разрез
			vector<int> minimumCut;
			vector<uint> newPlaneNormals = PlaneNormals;
			mincut(g, DIM, minimumCut);
			auto pos = find(minimumCut.begin(), minimumCut.end(), DIM - 2);
			if(pos != minimumCut.end()) {
				for(int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if(position != minimumCut.end()) { newPlaneNormals[kek] = BetaDir; } else { newPlaneNormals[kek] = AlphaDir; }
				}
			} else {
				for(int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if(position != minimumCut.end()) { newPlaneNormals[kek] = AlphaDir; } else { newPlaneNormals[kek] = BetaDir; }
				}
			}

			double EnergyValue_With_Hat = EnergyFunction(newPlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);
			if(EnergyValue_With_Hat < InitialEnergyValue) {
				PlaneNormals = newPlaneNormals;
				InitialEnergyValue = EnergyValue_With_Hat;
				success = 1;
			}
		}
	}
}

void ShowLSDLinesOnScreen(cv::Mat image, vector<Line> LSDLines) {
	int kk = 0;
	cv::Mat debi2 = image.clone();
	cv::cvtColor(debi2, debi2, CV_BGR2GRAY);
	cv::cvtColor(debi2, debi2, CV_GRAY2BGR);
	for(auto & l : LSDLines) {
		auto id = to_string(kk++);
		const cv::Scalar blaColor = cv::Scalar(RandomInt(0, 255), RandomInt(0, 255), RandomInt(0, 255), 255);
		auto begin = cv::Point(l.begin.x, l.begin.y);
		auto end = cv::Point(l.end.x, l.end.y);
		cv::line(debi2, begin, end, blaColor, 3);
		cv::circle(debi2, begin, 7, blaColor, -1);
		cv::putText(debi2, "<" + id, begin, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, CV_RGB(0, 0, 0), 1.5);
		cv::circle(debi2, end, 7, blaColor, -1);
		cv::putText(debi2, ">", end, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, CV_RGB(0, 0, 0), 1.5);
	}
	cv::imshow("LSD_Lines", debi2);
	cv::waitKey(0);
	cv::destroyWindow("LSD_Lines");
}

void ShowLineAtIterationStep(cv::Mat image, vector<int> indices, vector<cv::Point3f> beggienings, vector<cv::Point3f> endings, Line lineToJoin, Line l) {
	cv::Mat debi1 = image.clone();
	cv::cvtColor(debi1, debi1, CV_BGR2GRAY);
	cv::cvtColor(debi1, debi1, CV_GRAY2BGR);

	const cv::Scalar c1 = cv::Scalar(255, 0, 255, 255);
	const cv::Scalar c2 = cv::Scalar(255, 100, 10, 255);
	const cv::Scalar c3 = cv::Scalar(0, 200, 255, 255);
	for(auto i : indices) {
		cv::circle(debi1, cv::Point(beggienings[i].x, beggienings[i].y), 5, c2, -1);
		cv::line(debi1, cv::Point(beggienings[i].x, beggienings[i].y), cv::Point(endings[i].x, endings[i].y), c2, 2);
	}

	cv::line(debi1, cv::Point(l.begin.x, l.begin.y), cv::Point(l.end.x, l.end.y), c1, 2);
	cv::circle(debi1, cv::Point(l.end.x, l.end.y), 5, c1, -1);
	cv::line(debi1, cv::Point(lineToJoin.begin.x, lineToJoin.begin.y), cv::Point(lineToJoin.end.x, lineToJoin.end.y), c3, 1);

	cv::imshow("Iteration_Step", debi1);
	cv::waitKey(0);
	cv::destroyWindow("Iteration_Step");
}

void ShowJoinedLines(cv::Mat image, unordered_map<int, Line> lines) {
	cv::Mat copyImage3 = image.clone();
	for(auto & p : lines) {
		auto & l = p.second;
		auto id = to_string(p.first);
		const cv::Scalar blaColor = cv::Scalar(RandomInt(0, 255), RandomInt(0, 255), RandomInt(0, 255), 255);
		auto begin = cv::Point(l.begin.x, l.begin.y);
		auto end = cv::Point(l.end.x, l.end.y);
		cv::line(copyImage3, begin, end, blaColor, 3);
		cv::circle(copyImage3, begin, 7, blaColor, -1);
		cv::putText(copyImage3, "<" + id, begin, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, CV_RGB(0, 0, 0), 1.5);
		cv::circle(copyImage3, end, 7, blaColor, -1);
		cv::putText(copyImage3, id + ">", end, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, CV_RGB(0, 0, 0), 1.5);

	}
	//cv::imwrite("lines_all.jpg", copyImage3);
	cv::imshow("AllJoinedLines", copyImage3);
	cv::waitKey(0);
	cv::destroyWindow("AllJoinedLines");
}

void ExtendLines(cv::Mat image, vector<Line> LSDLines, unordered_map<int, Line> &output) {
	/// Present

	//memory layer 
	std::unordered_map<cv::Point3f, int, hashFunc> begginingsMemoryLayer(LSDLines.size());
	std::transform(std::begin(LSDLines), std::end(LSDLines),
				   std::inserter(begginingsMemoryLayer, begginingsMemoryLayer.end()),
				   [](const Line & l) {return std::make_pair(l.begin, l.index); });

	std::unordered_map<cv::Point3f, int, hashFunc> endingsMemoryLayer(LSDLines.size());
	std::transform(std::begin(LSDLines), std::end(LSDLines),
				   std::inserter(endingsMemoryLayer, endingsMemoryLayer.end()),
				   [](const Line & l) {return std::make_pair(l.end, l.index); });

	// points layer
	cv::flann::KDTreeIndexParams indexParams(3);
	using namespace boolinq;

	std::vector<cv::Point3f> beggienings;
	beggienings.reserve(LSDLines.size());
	for(auto &l : LSDLines) {
		beggienings.push_back(l.begin);
	}



	auto b = cv::Mat(beggienings).reshape(1);
	cv::flann::Index btree(b, indexParams);


	std::vector<cv::Point3f> endings;
	endings.reserve(LSDLines.size());
	for(auto l : LSDLines) {
		endings.push_back(l.end);
	}

	auto e = cv::Mat(endings).reshape(1);
	cv::flann::Index etree(b, indexParams);

	// Iteration step
	for(auto & l : LSDLines) {
		const int size = 5;
		std::vector<float> dist(size);
		std::vector<int> indices(size);
		auto found = btree.radiusSearch(std::vector<float>{l.end.x, l.end.y, l.end.z}, indices, dist, 15, size, cv::flann::SearchParams(100));
		if(found < size) {
			indices.resize(found);
		}
		std::vector<Line> lines;
		lines.reserve(found);

		int j = 0;
		for(auto &line : LSDLines) {
			
			auto it = find_if(indices.begin(), indices.end(), [&](const int& i)
			{
				return i == line.index;
			});
			if (it != indices.end() ) {
				lines.push_back(line);
			}
		}

		//TODO fix reorderting of one collection via values from another
		std::map<float, Line> m;
		for(auto & l : lines) {
			m[dist[j++]] = l;
		}
		int k = 0;
		for(auto & p : m) {
			lines[k++] = p.second;
		}

		if(lines.empty()) {
			continue;
		}

		
		Line lineToJoin;
		auto it = find_if(lines.begin(), lines.end(), [&](const Line & r) {
			return l.begin.z == r.begin.z;
		});
		if (it != lines.end()) {
			lineToJoin = *it;
		}
		else {
			continue;
		}
		

		if(l.index >= lineToJoin.index) { // only look behind
			l.index = lineToJoin.index;
		} else { // shall newer ocure!!!
			LSDLines[lineToJoin.index].index = l.index; //cant change LINQ copy 
		}

		///DEBUG_BEGIN
		//ShowLineAtIterationStep(image, indices, beggienings, endings, lineToJoin, l);
		///DEBUG_END
	}

	// correction step
	for(auto &l : LSDLines) {
		auto & line(output[l.index]);
		if(line.index == -1) {
			line = l;
		} else { // Todo check vertical line grouth!
			if(line.begin.x > l.begin.x) {
				line.begin = l.begin;
			}

			if(line.end.x < l.end.x) {
				line.end = l.end;
			}
			l.index = -2;
		}
	}
}

void ExtendLinesReversalMove(cv::Mat image, vector<Line> LSDLines, unordered_map<int, Line> &output) {
	/// Present

	//memory layer 
	std::unordered_map<cv::Point3f, int, hashFunc> begginingsMemoryLayer(LSDLines.size());
	std::transform(std::begin(LSDLines), std::end(LSDLines),
				   std::inserter(begginingsMemoryLayer, begginingsMemoryLayer.end()),
				   [](const Line & l) {return std::make_pair(l.begin, l.index); });

	std::unordered_map<cv::Point3f, int, hashFunc> endingsMemoryLayer(LSDLines.size());
	std::transform(std::begin(LSDLines), std::end(LSDLines),
				   std::inserter(endingsMemoryLayer, endingsMemoryLayer.end()),
				   [](const Line & l) {return std::make_pair(l.end, l.index); });

	// points layer
	cv::flann::KDTreeIndexParams indexParams(3);
	using namespace boolinq;

	std::vector<cv::Point3f> beggienings;
	beggienings.reserve(LSDLines.size());
	for(auto &l : LSDLines) {
		beggienings.push_back(l.begin);
	}



	auto b = cv::Mat(beggienings).reshape(1);
	cv::flann::Index btree(b, indexParams);


	std::vector<cv::Point3f> endings;
	endings.reserve(LSDLines.size());
	for(auto l : LSDLines) {
		endings.push_back(l.end);
	}

	auto e = cv::Mat(endings).reshape(1);
	cv::flann::Index etree(e, indexParams);

	// Iteration step
	for(auto & l : LSDLines) {
		const int size = 5;
		std::vector<float> dist(size);
		std::vector<int> indices(size);
		auto found = etree.radiusSearch(std::vector<float>{l.begin.x, l.begin.y, l.begin.z}, indices, dist, 15, size, cv::flann::SearchParams(100));
		if(found < size) {
			indices.resize(found);
		}
		std::vector<Line> lines;
		lines.reserve(found);

		int j = 0;
		for(auto &line : LSDLines) {

			auto it = find_if(indices.begin(), indices.end(), [&](const int& i) {
				return i == line.index;
			});
			if(it != indices.end()) {
				lines.push_back(line);
			}
		}

		//TODO fix reorderting of one collection via values from another
		std::map<float, Line> m;
		for(auto & l : lines) {
			m[dist[j++]] = l;
		}
		int k = 0;
		for(auto & p : m) {
			lines[k++] = p.second;
		}

		if(lines.empty()) {
			continue;
		}


		Line lineToJoin;
		auto it = find_if(lines.begin(), lines.end(), [&](const Line & r) {
			return l.end.z == r.end.z;
		});
		if(it != lines.end()) {
			lineToJoin = *it;
		} else {
			continue;
		}


		if(l.index >= lineToJoin.index) { // only look behind
			l.index = lineToJoin.index;
		} else { // shall newer ocure!!!
			LSDLines[lineToJoin.index].index = l.index; //cant change LINQ copy 
		}

		///DEBUG_BEGIN
		//ShowLineAtIterationStep(image, indices, beggienings, endings, lineToJoin, l);
		///DEBUG_END
	}

	// correction step
	for(auto &l : LSDLines) {
		auto & line(output[l.index]);
		if(line.index == -1) {
			line = l;
		} else { // Todo check vertical line grouth!
			if(line.begin.x > l.begin.x) {
				line.begin = l.begin;
			}

			if(line.end.x < l.end.x) {
				line.end = l.end;
			}
			l.index = -2;
		}
	}
}


///INTERSECTION_DETECTION_BEGIN
double crossProduct(cv::Point3f a, cv::Point3f b) {
	return a.x * b.y - b.x * a.y;
}

bool doBoundingBoxesIntersect(Line a, Line b) {
	return a.begin.x <= b.end.x && a.end.x >= b.begin.x && a.begin.y <= b.end.y
		&& a.end.y >= b.begin.y;
}

bool isPointOnLine(Line a, cv::Point3f b, double EPSILON) {
	Line aTmp = Line(cv::Point3f(0, 0, 0), cv::Point3f(
		a.end.x - a.begin.x, a.end.y - a.begin.y, 0));
	cv::Point3f bTmp = cv::Point3f(b.x - a.begin.x, b.y - a.begin.y, 0);
	double r = crossProduct(aTmp.end, bTmp);
	return abs(r) < EPSILON;
}

bool isPointRightOfLine(Line a, cv::Point3f b) {
	Line aTmp = Line(cv::Point3f(0, 0, 0), cv::Point3f(
		a.end.x - a.begin.x, a.end.y - a.begin.y, 0));
	cv::Point3f bTmp = cv::Point3f(b.x - a.begin.x, b.y - a.begin.y, 0);
	return crossProduct(aTmp.end, bTmp) < 0;
}

bool lineSegmentTouchesOrCrossesLine(Line a, Line b, double EPSILON) {
	return isPointOnLine(a, b.begin, EPSILON)
		|| isPointOnLine(a, b.end, EPSILON)
		|| (isPointRightOfLine(a, b.begin) ^ isPointRightOfLine(a,
			b.end));
}

bool doLinesIntersect(Line a, Line b, double EPSILON) {
	return doBoundingBoxesIntersect(a, b)
		&& lineSegmentTouchesOrCrossesLine(a, b, EPSILON)
		&& lineSegmentTouchesOrCrossesLine(b, a, EPSILON);
}

void getLinePairs(unordered_map<int, Line> lines, vector<PairOfTwoLines>& outputVec) {
	double EPSILON = 0.000001;
	for(auto & firstLine : lines) {
		for(auto & secondLine : lines) {
			if (doLinesIntersect(firstLine.second, secondLine.second, EPSILON)) {
				PairOfTwoLines tempPair = PairOfTwoLines(firstLine.first, secondLine.first);
				outputVec.push_back(tempPair);
			}
		}
	}
}
///INTERSECTION_DETECTION_END

int main() {
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
	if(maxRes > ResizeIfMoreThan) {
		float scaleFactor = float(ResizeIfMoreThan) / maxRes;
		cv::Size size = cv::Size(image.cols * scaleFactor, image.rows * scaleFactor);
		cv::resize(image, image, size);
	}

	//Границы кадра
	uint maxX = image.cols;
	uint maxY = image.rows;

	//Фокальное расстояние
	double temp_focal = maxX * focal_length / sensor_width;
	uint f = (uint)temp_focal; //фокальное расстояние в пикселях

	//LSD
	int numLinesDetected;
	double* LsdLinesArray = DoLSD(image, numLinesDetected);
	cout << "Number of LSD lines detected: " << numLinesDetected << endl;

	//Копируем точки из массива в вектор с учетом порядка
	vector<cv::Point3f> LsdLinesVector; //элемент с номером 2*i дает нам точку начала i-ой линии, элемент с номером 2*i + 1 дает нам точку конца i-ой линии
	for(int i = 0; i < numLinesDetected; i++) {
		double x1 = RoundTo(LsdLinesArray[7 * i]);
		double y1 = RoundTo(LsdLinesArray[7 * i + 1]);
		double x2 = RoundTo(LsdLinesArray[7 * i + 2]);
		double y2 = RoundTo(LsdLinesArray[7 * i + 3]);
		cv::Point3f point_one = cv::Point3f(x1, y1, 0);
		cv::Point3f point_two = cv::Point3f(x2, y2, 0);
		if(x1 < x2) { // todo use epsilon!
			LsdLinesVector.push_back(point_one);
			LsdLinesVector.push_back(point_two);
		} else if(x2 < x1) {
			LsdLinesVector.push_back(point_two);
			LsdLinesVector.push_back(point_one);
		} else if(y1 < y2) {
			LsdLinesVector.push_back(point_one);
			LsdLinesVector.push_back(point_two);
		} else {
			LsdLinesVector.push_back(point_two);
			LsdLinesVector.push_back(point_one);
		}
		auto & a(LsdLinesVector[i * 2 + 1]);
		auto & b(LsdLinesVector[i * 2]);
		auto dir = a.y - b.y;
		dir /= abs(dir);
		a.z = dir;
		b.z = dir;
	}

	//todo: split into ~3 line long seprate member functions
	// Sorted left->right lines layer
	vector<Line> LSDLines = [](auto LsdLines) //todo extract method
	{
		vector<Line> result(LsdLines.size() / 2);
		for(int i = 0; i < result.size(); ++i) {
			result[i] = Line(LsdLines[i * 2], LsdLines[i * 2 + 1], i);
		}
		return result;
	}(LsdLinesVector);

	unordered_map<int, Line> lines;

	///DEBUG_BEGIN
	ShowLSDLinesOnScreen(image, LSDLines);
	///DEBUG_END
	auto t1 = chrono::high_resolution_clock::now().time_since_epoch();

	for(int i = 0; i < 2; ++i) {
		lines = unordered_map<int, Line>();
		ExtendLines(image, LSDLines, lines);
		//ExtendLinesReversalMove(image, LSDLines, lines); //обратный ход
		LSDLines = vector<Line>();
		LSDLines.resize(lines.size());
		cout << "reduced on iteration " << i << " lines count: " << lines.size() << endl;
		for(auto &p : lines) {
			//noise filter
			if (p.second.norm() > (5*maxX/100)) {
				LSDLines.push_back(p.second);
			}
		}
	}
	auto dt = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch() - t1).count();
	cout << "it took " << dt << endl; // 247ms for 10 iterations, 10ms for one iteration, 35ms for 2 on one core

	///DEBUG_BEGIN
	ShowJoinedLines(image, lines);
	///DEBUG_END


	//Getting intersections
	vector<PairOfTwoLines> LinePairsVector;
	getLinePairs(lines, LinePairsVector);
	cout << "Number of line pairs: " << LinePairsVector.size() << endl;

	return 0;

	vector<cv::Point3f> ExtendedLinesVector;
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

	return 0;


	/// Past
	//Выделяем все замкнутые области
	vector<vector<uint> > PolygonsVector;
	getPolygons(numLinesDetected, ExtendedLinesVector, AngleTolerance, step, radius, PolygonsVector);

	/*
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
	*/
	system("pause");
	return 0;
}
