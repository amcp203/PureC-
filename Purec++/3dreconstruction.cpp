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
int RoundTo(double x) {
	int y = floor(x);
	if ((x - y) >= 0.5)
		y++;
	return y;
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

void ExtendLines(int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f>& ExtendedLinesVector, vector<PairOfTwoLines>& LinePairsVector, double threshold) {
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

void assignDirections(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<cv::Point3f> VanishingPoints, vector<int>& output) {
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

		int result;
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

int main()
{	
	clock_t tStart = clock();
	srand(time(0)); //чтобы последовательность рандомных чисел была всегда уникальна при новом запуске

	//Переменные для настройки
	double focal_length = 4; //фокальное расстояние в mm
	double sensor_width = 4.59;
	double ExtendThreshold = 0.01; //порог отклонения линии (для удлинения)
	double countInlierThreshold = 0.0001; //если квадрат скалярного произведения двух линий меньше этого числа, то мы считаем эти линии ортогональными
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
		double x1 = LsdLinesArray[7 * i];
		double y1 = LsdLinesArray[7 * i + 1];
		double x2 = LsdLinesArray[7 * i + 2];
		double y2 = LsdLinesArray[7 * i + 3];
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
	ExtendLines(numLinesDetected, LsdLinesVector, ExtendedLinesVector, LinePairsVector, ExtendThreshold);
	cout << "Number of line pairs: " << LinePairsVector.size() << endl;
	
	//RANSAC (находим углы альфа и бэта)
	uint maxRansacTrials = (uint)LinePairsVector.size(); //максимальное количество итерация алгоритма RANSAC
	dlib::matrix<double, 0, 1> solution = RANSAC(maxRansacTrials, countInlierThreshold, f, LinePairsVector, ExtendedLinesVector);
	cout << "angle Alpha (in radians): " << solution(0) << endl << "angle Beta (in radians): " << solution(1) << endl;

	//Получаем vanishing points
	vector<cv::Point3f> VanishingPoints;
	getVanishingPoints(solution(0), solution(1), f, VanishingPoints);

	//Находим направление каждой линии
	vector<int> DirectionsOfLines; //элемент с номером i означает направление i-ой линии (0=x, 1=y, 2=z, 3=xy, 4=xz, 5=yz)
	assignDirections(numLinesDetected, ExtendedLinesVector, VanishingPoints, DirectionsOfLines);





	if (debug) {
		write_eps(LsdLinesArray, numLinesDetected, 7, "test.eps", maxX, maxY, 1);
		double * test = (double *)malloc(maxX * maxY * sizeof(double));
		for (int i = 0; i < numLinesDetected; i++) {
			test[7 * i] = ExtendedLinesVector[2 * i].x;
			test[7 * i + 1] = ExtendedLinesVector[2 * i].y;
			test[7 * i + 2] = ExtendedLinesVector[2 * i + 1].x;
			test[7 * i + 3] = ExtendedLinesVector[2 * i + 1].y;
			test[7 * i + 4] = 1;
			test[7 * i + 5] = 0.125;
			test[7 * i + 6] = 15;
		}
		write_eps(test, numLinesDetected, 7, "test_extend.eps", maxX, maxY, 1);
	}

	system("pause");
	return 0;
}

