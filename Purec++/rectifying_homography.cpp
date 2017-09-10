#include <opencv2/core/core.hpp>
#include <vector>
#include "dlib/optimization.h"
#include <rectifying_homography.h>
#define _USE_MATH_DEFINES
#include <math.h>


typedef uint32_t uint;
using namespace std;

//Intrinsic matrix
cv::Mat K_inv(double f) {
	double temp[3][3] = { { f, 0, 0 },{ 0, f, 0 },{ 0, 0, 1 } };
	cv::Mat k = cv::Mat(3, 3, CV_64F, temp);
	return k.inv();
}


//»щет номер линии по координатам точки
int FindIndexOfLine(vector<cv::Point3f> vec, cv::Point3f point) {
	auto position = find_if(vec.begin(), vec.end(), [&](const cv::Point3f& a) { return a.x == point.x && a.y == point.y; });
	if (position != vec.end()) {
		int index = (position - vec.begin()) / 2;
		return index;
	}
	return -1;
}

cost_function::cost_function(double f, cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, cv::Point3f e, cv::Point3f ff, cv::Point3f g, cv::Point3f h) {
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

double cost_function::operator()(const dlib::matrix<double, 0, 1>& arg) const {
	double summ = 0;

	double r_x_array[3][3] = { { 1, 0, 0 },{ 0, cos(arg(0)), -sin(arg(0)) },{ 0, sin(arg(0)), cos(arg(0)) } };
	cv::Mat R_x = cv::Mat(3, 3, CV_64F, r_x_array);
	double r_y_array[3][3] = { { cos(arg(1)), 0, sin(arg(1)) },{ 0, 1, 0 },{ -sin(arg(1)), 0, cos(arg(1)) } };
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

//ћинимизаци€ cost function дл€ двух пар пересекающихс€ линий
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

//—читает сколько пар линий стало ортогональными при найденных alpha и beta
uint countInlierScore(dlib::matrix<double, 0, 1> solution, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint score = 0;
	cv::Mat K_inverted = K_inv(f);
	double temp[3][3] = { { 1, 0, 0 },{ 0, cos(solution(0)), -sin(solution(0)) },{ 0, sin(solution(0)), cos(solution(0)) } };
	cv::Mat R_x = cv::Mat(3, 3, CV_64F, temp);
	double temp2[3][3] = { { cos(solution(1)), 0, sin(solution(1)) },{ 0, 1, 0 },{ -sin(solution(1)), 0, cos(solution(1)) } };
	cv::Mat R_y = cv::Mat(3, 3, CV_64F, temp2);
	cv::Mat H = R_y * R_x * K_inverted;
	cv::Mat H_t = H.t();
	for (int i = 0; i < LinePairsVector.size(); i++) {
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
		if (summ <= threshold) { score++; }
	}
	return score;
}

//јлгоритм RANSAC (берет рандомные пары линий, делает минимизацию и считает сколько пар стало ортогональными, потом выбирает лучшее решение
dlib::matrix<double, 0, 1> RANSAC_(uint maxTrials, double threshold, double f, vector<PairOfTwoLines> LinePairsVector, vector<cv::Point3f> ExtendedLinesVector) {
	uint counter = 0;
	uint bestScore = 0;
	dlib::matrix<double, 0, 1> bestSolution(2);
	while (counter < maxTrials) {
		uint first_index = RandomInt(0, LinePairsVector.size() - 1);
		uint second_index = first_index;
		while (second_index == first_index) { second_index = RandomInt(0, LinePairsVector.size() - 1); }
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
