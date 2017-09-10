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

#include <tools.h>
#include <depth_estimation.h>
#include <rectifying_homography.h>

//OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

//LSD
#include "lsd/lsd.h"
#include "boolinq/boolinq.h"

typedef uint32_t uint;
using namespace std;

/**
*  \brief Automatic brightness and contrast optimization with optional histogram clipping
*  \param [in]src Input image GRAY or BGR or BGRA
*  \param [out]dst Destination image
*  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
*  \note In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent = 0) {

	CV_Assert(clipHistPercent >= 0);
	CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

	int histSize = 256;
	float alpha, beta;
	double minGray = 0, maxGray = 0;

	//to calculate grayscale histogram
	cv::Mat gray;
	if (src.type() == CV_8UC1) gray = src;
	else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
	else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
	if (clipHistPercent == 0) {
		// keep full available range
		cv::minMaxLoc(gray, &minGray, &maxGray);
	}
	else {
		cv::Mat hist; //the grayscale histogram

		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// calculate cumulative distribution from the histogram
		std::vector<float> accumulator(histSize);
		accumulator[0] = hist.at<float>(0);
		for (int i = 1; i < histSize; i++) {
			accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
		}

		// locate points that cuts at required value
		float max = accumulator.back();
		clipHistPercent *= (max / 100.0); //make percent as absolute
		clipHistPercent /= 2.0; // left and right wings
								// locate left cut
		minGray = 0;
		while (accumulator[minGray] < clipHistPercent)
			minGray++;

		// locate right cut
		maxGray = histSize - 1;
		while (accumulator[maxGray] >= (max - clipHistPercent))
			maxGray--;
	}

	// current range
	float inputRange = maxGray - minGray;

	alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
	beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

										 // Apply brightness and contrast normalization
										 // convertTo operates with saurate_cast
	src.convertTo(dst, -1, alpha, beta);

	// restore alpha channel from source 
	if (dst.type() == CV_8UC4) {
		int from_to[] = { 3, 3 };
		cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
	}
	return;
}

//Write LSD Lines to .eps file
void write_eps(double* segs, int n, int dim, char* filename, int xsize, int ysize, double width) {
	FILE* eps;
	int i;

	/* open file */
	if (strcmp(filename, "-") == 0)
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
	for (i = 0; i < n; i++) {
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
	auto im = image.clone();
	cv::cvtColor(im, grayscaleMat, CV_BGR2GRAY);


	/////////////////////////
	Canny(grayscaleMat, grayscaleMat, 50, 200, 3, true);
	vector<cv::Vec4i> lines;
	cv::HoughLinesP(grayscaleMat, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++) {
		cv::Vec4i l = lines[i];
		auto begin = cv::Point(l[0], l[1]);
		auto end = cv::Point(l[2], l[3]);
		line(im, begin, end, cv::Scalar(0, 0, 255), 1, CV_AA);
		cv::circle(im, begin, 3, cv::Scalar(0, 200, 0), -1);
		cv::circle(im, end, 3, cv::Scalar(200, 0, 0), -1);
	}
	cout << "Hough Lines" << lines.size() << endl;
	//imshow("cany edges", grayscaleMat);

	//imshow("Hough Lines", im);
	//cv::waitKey(0);
	/////////////////////////////////////////////

	double* pgm_image;
	pgm_image = (double *)malloc(image.cols * image.rows * sizeof(double));
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
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

class Line {
public:
	cv::Point3f begin;
	cv::Point3f end;
	int index;

	Line(cv::Point3f a = cv::Point3f(-1, -1, -1), cv::Point3f b = cv::Point3f(-1, -1, -1), int index = -1) :
		begin(a), end(b), index(index) {};
	double norm() {
		double norm = sqrt((end.x - begin.x)*(end.x - begin.x) + (end.y - begin.y)*(end.y - begin.y));
		return RoundTo(norm);
	};
};




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

	double kArray[3][3] = { { f, 0, 0 },{ 0, f, 0 },{ 0, 0, 1 } };
	cv::Mat K_matrix = cv::Mat(3, 3, CV_64F, kArray);
	double r_x_array[3][3] = { { 1, 0, 0 },{ 0, cos(alpha), -sin(alpha) },{ 0, sin(alpha), cos(alpha) } };
	cv::Mat Rx = cv::Mat(3, 3, CV_64F, r_x_array);
	double r_y_array[3][3] = { { cos(beta), 0, sin(beta) },{ 0, 1, 0 },{ -sin(beta), 0, cos(beta) } };
	cv::Mat Ry = cv::Mat(3, 3, CV_64F, r_y_array);

	cv::Mat R = Ry * Rx;
	cv::Mat R_t = R.t();

	double XVectorArray[3][1] = { { 0 },{ 0 },{ 1 } };
	cv::Mat XMat = cv::Mat(3, 1, CV_64F, XVectorArray);
	double YVectorArray[3][1] = { { 0 },{ 1 },{ 0 } };
	cv::Mat YMat = cv::Mat(3, 1, CV_64F, YVectorArray);
	double ZVectorArray[3][1] = { { 1 },{ 0 },{ 0 } };
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
	for (int i = 0; i < numLinesDetected; i++) {
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
		if (cosX <= 0.9 && cosY <= 0.9 && abs(cosX - cosY) < 0.01) { result = 3; }
		else if (cosX <= 0.9 && cosZ <= 0.9 && abs(cosX - cosZ) < 0.01) { result = 4; }
		else if (cosY <= 0.9 && cosZ <= 0.9 && abs(cosY - cosZ) < 0.01) { result = 5; }
		else if (cosX <= 0.9) { result = 0; }
		else if (cosY <= 0.9) { result = 1; }
		else
			if (cosZ <= 0.9) { result = 2; }
		output.push_back(result);
	}
}

void getPolygons(int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, double AngleTolerance, double step, double radius, vector<vector<uint> >& PolygonsVector, bool debugFlag, cv::Mat image, double distanceEpsilon, double AngleEpsilon) {
	//Выделение групп параллельных линий
	vector<vector<uint> > ParallelLineGroups;
	vector<uint> FirstGroup;
	ParallelLineGroups.push_back(FirstGroup);
	vector<double> GroupCoefficients;
	GroupCoefficients.push_back(0);
	vector<uint> VerticalLinesGroup;
	for (int i = 0; i < numLinesDetected; i++) {
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		if (endPoint.x - beginPoint.x != 0) {
			double AngleCoef = (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x);
			bool Found = false;
			for (int j = 0; j < GroupCoefficients.size(); j++) {
				if (abs(AngleCoef - GroupCoefficients[j]) < AngleTolerance) {
					ParallelLineGroups[j].push_back(i);
					Found = true;
				}
			}
			if (!Found) {
				GroupCoefficients.push_back(AngleCoef);
				vector<uint> Group;
				Group.push_back(i);
				ParallelLineGroups.push_back(Group);
			}
		}
		else { VerticalLinesGroup.push_back(i); }
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
	for (int i = 0; i < numLinesDetected; i++) {
		uint totalPoints = 0;
		cv::Point3f beginPoint = ExtendedLinesVector[2 * i];
		cv::Point3f endPoint = ExtendedLinesVector[2 * i + 1];
		double currentX = beginPoint.x + step;
		while (currentX <= endPoint.x) {
			double currentY = (currentX - beginPoint.x) * (endPoint.y - beginPoint.y) / (endPoint.x - beginPoint.x) + beginPoint.y;
			cv::Point3f point = cv::Point3f(currentX, currentY, 0);
			pointsForSearch.push_back(point);
			PointIndexes.push_back(i); //запоминаем какая точка к какой линии принадлежит
			totalPoints++;
			currentX += step;
		}
		if (totalPoints != 0) { LineScoreVector[i].totalPoints = totalPoints; }
	}

	//Поиск линий в Octree
	cv::Octree tree = cv::Octree(pointsForSearch);
	for (int i = 0; i < ParallelLineGroups.size(); i++) {
		vector<uint> Group = ParallelLineGroups[i];
		if (Group.size() >= 2) {
			for (int j = 0; j < Group.size(); j++) {

				///TEST VERSION Begin
				vector<NearestLine> vect;
				uint FirstIndex = Group[j];
				cv::Vec3f firstLine = ExtendedLinesVector[2 * FirstIndex + 1] - ExtendedLinesVector[2 * FirstIndex];
				cv::Point3f firstMiddlePoint = cv::Point3f((ExtendedLinesVector[2 * FirstIndex + 1].x + ExtendedLinesVector[2 * FirstIndex].x) / 2, (ExtendedLinesVector[2 * FirstIndex + 1].y + ExtendedLinesVector[2 * FirstIndex].y) / 2, 0);

				for (int key = j + 1; key < Group.size(); key++) {
					uint SecondIndex = Group[key];
					cv::Vec3f secondLine = ExtendedLinesVector[2 * SecondIndex + 1] - ExtendedLinesVector[2 * SecondIndex];
					cv::Point3f secondMiddlePoint = cv::Point3f((ExtendedLinesVector[2 * SecondIndex + 1].x + ExtendedLinesVector[2 * SecondIndex].x) / 2, (ExtendedLinesVector[2 * SecondIndex + 1].y + ExtendedLinesVector[2 * SecondIndex].y) / 2, 0);
					cv::Vec3f middleVector = cv::Vec3f(firstMiddlePoint - secondMiddlePoint);
					double distance = cv::norm(middleVector, cv::NORM_L2);
					if (distance > distanceEpsilon) {
						NearestLine current = NearestLine(key, distance);
						vect.push_back(current);
					}

				}

				sort(vect.begin(), vect.end(), [&](NearestLine a, NearestLine b) {
					return a.dist < b.dist;
				});
				///TEST VERSION END

				//for(int k = j + 1; k < Group.size(); k++) { 
				if (vect.size() > 0) {
					//uint FirstIndex = Group[j];
					uint SecondIndex = Group[vect[0].index];
					cv::Point3f leftLineBeginPoint;
					cv::Point3f leftLineEndPoint;
					cv::Point3f rightLineBeginPoint;
					cv::Point3f rightLineEndPoint;
					/*
					//Вывод линий на экран
					cv::Mat debImage = image.clone();
					const cv::Scalar blaColor = cv::Scalar(0, 0, 255, 255);
					for (int ka = 0; ka < Group.size(); ka++) {
					uint index = Group[ka];
					auto begin = cv::Point(ExtendedLinesVector[2 * index].x, ExtendedLinesVector[2 * index].y);
					auto end = cv::Point(ExtendedLinesVector[2 * index + 1].x, ExtendedLinesVector[2 * index + 1].y);
					cv::line(debImage, begin, end, blaColor, 3);
					cv::circle(debImage, begin, 7, blaColor, -1);
					cv::circle(debImage, end, 7, blaColor, -1);

					}

					const cv::Scalar blaColorMain = cv::Scalar(255, 0, 0, 255);
					auto beginF = cv::Point(ExtendedLinesVector[2 * FirstIndex].x, ExtendedLinesVector[2 * FirstIndex].y);
					auto endF = cv::Point(ExtendedLinesVector[2 * FirstIndex + 1].x, ExtendedLinesVector[2 * FirstIndex + 1].y);
					auto beginS = cv::Point(ExtendedLinesVector[2 * SecondIndex].x, ExtendedLinesVector[2 * SecondIndex].y);
					auto endS = cv::Point(ExtendedLinesVector[2 * SecondIndex + 1].x, ExtendedLinesVector[2 * SecondIndex + 1].y);
					cv::line(debImage, beginF, endF, blaColorMain, 3);
					cv::circle(debImage, beginF, 7, blaColorMain, -1);
					cv::circle(debImage, endF, 7, blaColorMain, -1);
					cv::line(debImage, beginS, endS, blaColorMain, 3);
					cv::circle(debImage, beginS, 7, blaColorMain, -1);
					cv::circle(debImage, endS, 7, blaColorMain, -1);

					cv::imshow("NearestLines", debImage);
					cv::waitKey(0);
					cv::destroyWindow("NearestLines");*/

					//Смотрим какая линия левее
					if (ExtendedLinesVector[2 * SecondIndex + 1].x >= ExtendedLinesVector[2 * FirstIndex + 1].x) {
						leftLineBeginPoint = ExtendedLinesVector[2 * FirstIndex];
						leftLineEndPoint = ExtendedLinesVector[2 * FirstIndex + 1];
						rightLineBeginPoint = ExtendedLinesVector[2 * SecondIndex];
						rightLineEndPoint = ExtendedLinesVector[2 * SecondIndex + 1];
					}
					else {
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
					for (int g = 0; g < centersForSearchBegin.size(); g++) {
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchBegin[g], radius, foundedPoints);
						for (int h = 0; h < foundedPoints.size(); h++) {
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a) { //идентифицируем к какой линии принадлежит найденная точка
								return a.x == tempPoint.x && a.y == tempPoint.y;
							});
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if (index != FirstIndex && index != SecondIndex) {
								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(beginLineExpected.dot(line) / (cv::norm(line) * cv::norm(beginLineExpected)));
								if (debugFlag) {
									cv::Vec3f lineA = leftLineEndPoint - leftLineBeginPoint;
									double cos = abs(lineA.dot(line) / (cv::norm(line) * cv::norm(lineA)));
									if (cosAngle < AngleEpsilon) { currentBeginScores[index].goodPoints++; }
								}
								else {
									if (cosAngle > 0.95) { //фильтрация совсем трешовых вариантов
										currentBeginScores[index].goodPoints++;
									}
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
					while (currentX <= rightLineEndPoint.x) {
						double currentY = (currentX - leftLineEndPoint.x) * (rightLineEndPoint.y - leftLineEndPoint.y) / (rightLineEndPoint.x - leftLineEndPoint.x) + leftLineEndPoint.y;
						cv::Point3f point = cv::Point3f(currentX, currentY, 0);
						centersForSearchEnd.push_back(point);
						currentX += step / 2;
					}
					for (int g = 0; g < centersForSearchEnd.size(); g++) {
						vector<cv::Point3f> foundedPoints;
						tree.getPointsWithinSphere(centersForSearchEnd[g], radius, foundedPoints);
						for (int h = 0; h < foundedPoints.size(); h++) {
							cv::Point3f tempPoint = foundedPoints[h];
							auto position = find_if(pointsForSearch.begin(), pointsForSearch.end(), [&](const cv::Point3f& a) { return a.x == tempPoint.x && a.y == tempPoint.y; });
							uint indexInVector = position - pointsForSearch.begin();
							uint index = PointIndexes[indexInVector];
							if (index != FirstIndex && index != SecondIndex) {

								cv::Point3f beginPointFound = ExtendedLinesVector[index * 2];
								cv::Point3f endPointFound = ExtendedLinesVector[index * 2 + 1];
								cv::Vec3f line = endPointFound - beginPointFound;
								double cosAngle = abs(endLineExpected.dot(line) / (cv::norm(line) * cv::norm(endLineExpected)));
								if (debugFlag) {
									cv::Vec3f lineA = leftLineEndPoint - leftLineBeginPoint;
									double cos = abs(lineA.dot(line) / (cv::norm(line) * cv::norm(lineA)));
									if (cosAngle < AngleEpsilon) { currentEndScores[index].goodPoints++; }

								}
								else {
									if (cosAngle > 0.95) { currentEndScores[index].goodPoints++; }
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

void getNumbersOfPixels(cv::Mat image, vector<vector<uint> > PolygonsVector, int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<int>& NumberOfPixelsInArea) {
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
	for (int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> linesA = PolygonsVector[i];

		for (int j = i + 1; j < PolygonsVector.size(); j++) {
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
		if (direction == DirectionsOfLines[index]) { result += Wocc * lineSupportingScore(lines, index, numLinesDetected, LsdLinesVector, ExtendedLinesVector); }
	}
	return result;
}

double dataCost(vector<uint> lines, uint direction, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, double Warea) {
	double result = 0;
	if (direction == 0) { result = -Warea * (DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 5, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); }
	else if (direction == 1) { result = -Warea * (DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 2, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 4, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); }
	else
		if (direction == 2) { result = -Warea * (DirectionScore(lines, 0, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 1, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines) + DirectionScore(lines, 3, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); }
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
		double Wl = 1 - 0.9 * (LSD_length / CommonLine_length);
		double Wdir;
		if (directionA != DirectionsOfLines[commonLine] && directionB != DirectionsOfLines[commonLine]) { Wdir = 0.1; }
		else { Wdir = 1.0; }
		return Wdir * Wl * Csmooth;
	}
	return 0;

}

double EnergyFunction(vector<uint> PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	double dataCostValue = 0;
	double smoothnessCostValue = 0;
	for (int i = 0; i < PolygonsVector.size(); i++) {
		vector<uint> lines = PolygonsVector[i];
		uint normalDirection = PlaneNormals[i];
		double Warea = 1 + (double)NumberOfPixelsInArea[i] / 100;
		dataCostValue += dataCost(lines, normalDirection, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, Warea);
		for (int j = i; j < PolygonsVector.size(); j++) { if (i != j && PolygonIntersections[i][j] == 1) { smoothnessCostValue += smoothnessCost(lines, PolygonsVector[j], normalDirection, PlaneNormals[j], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
	}
	return dataCostValue + smoothnessCostValue;
}

void mincut(int** g, int n, vector<int>& best_cut) {
	const int MAXN = 5000;
	int best_cost = 1000000000;
	vector<int> v[MAXN];
	for (int i = 0; i < n; ++i)
		v[i].assign(1, i);
	int w[MAXN];
	bool exist[MAXN], in_a[MAXN];
	memset(exist, true, sizeof exist);
	for (int ph = 0; ph < n - 1; ++ph) {
		memset(in_a, false, sizeof in_a);
		memset(w, 0, sizeof w);
		for (int it = 0, prev; it < n - ph; ++it) {
			int sel = -1;
			for (int i = 0; i < n; ++i)
				if (exist[i] && !in_a[i] && (sel == -1 || w[i] > w[sel]))
					sel = i;
			if (it == n - ph - 1) {
				if (w[sel] < best_cost)
					best_cost = w[sel], best_cut = v[sel];
				v[prev].insert(v[prev].end(), v[sel].begin(), v[sel].end());
				for (int i = 0; i < n; ++i)
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

void getNormals(vector<uint>& PlaneNormals, int numLinesDetected, vector<cv::Point3f> LsdLinesVector, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, vector<int> NumberOfPixelsInArea, bool** PolygonIntersections) {
	for (int i = 0; i < PolygonsVector.size(); i++) {
		uint randomNormalDirection = RandomInt(0, 2);
		PlaneNormals.push_back(randomNormalDirection);
	}
	double InitialEnergyValue = EnergyFunction(PlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);

	bool success = 1;
	while (success) {
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
			int** g = new int *[DIM];;
			for (int j = 0; j < DIM; j++) { g[j] = new int[DIM]; }
			g[DIM - 1][DIM - 1] = 0;
			g[DIM - 2][DIM - 2] = 0;
			g[DIM - 1][DIM - 2] = 0;
			g[DIM - 2][DIM - 1] = 0;

			for (int i = 0; i < DIM - 2; i++) {
				if (PlaneNormals[i] != AlphaDir && PlaneNormals[i] != BetaDir) {
					int* zeroArray = new int[DIM];
					memset(zeroArray, 0, sizeof(zeroArray));
					g[i] = zeroArray;
					for (int k = 0; k < DIM; k++) { g[k][i] = 0; }
				}
				else {
					int dataCostAlpha = dataCost(PolygonsVector[i], AlphaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);
					int dataCostBeta = dataCost(PolygonsVector[i], BetaDir, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, NumberOfPixelsInArea[i]);;
					double smoothnessCostAlpha = 0;
					for (int it = 0; it < PolygonsVector.size(); it++) { if (PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) { smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], AlphaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
					double smoothnessCostBeta = 0;
					for (int it = 0; it < PolygonsVector.size(); it++) { if (PolygonIntersections[i][it] == 1 && PlaneNormals[it] != AlphaDir && PlaneNormals[it] != BetaDir) { smoothnessCostAlpha += smoothnessCost(PolygonsVector[i], PolygonsVector[it], BetaDir, PlaneNormals[it], LsdLinesVector, ExtendedLinesVector, DirectionsOfLines); } }
					g[i][DIM - 2] = dataCostAlpha + (int)RoundTo(smoothnessCostAlpha); //tlinkA
					g[i][DIM - 1] = dataCostBeta + (int)RoundTo(smoothnessCostBeta); // tlinkB
					for (int j = i; j < DIM - 2; j++) {

						if (i == j || PolygonIntersections[i][j] == 0) { g[i][j] = 0; }
						else {
							if (PlaneNormals[j] == AlphaDir || PlaneNormals[j] == BetaDir) { g[i][j] = (int)RoundTo(smoothnessCost(PolygonsVector[i], PolygonsVector[j], AlphaDir, BetaDir, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines)); }
							else { g[i][j] = 0; }
						}
					}
				}
			}
			for (int i = 0; i < DIM; i++) { for (int j = 0; j < i; j++) { g[i][j] = g[j][i]; } }

			//делаем минимальный разрез
			vector<int> minimumCut;
			vector<uint> newPlaneNormals = PlaneNormals;
			mincut(g, DIM, minimumCut);
			auto pos = find(minimumCut.begin(), minimumCut.end(), DIM - 2);
			if (pos != minimumCut.end()) {
				for (int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if (position != minimumCut.end()) { newPlaneNormals[kek] = BetaDir; }
					else { newPlaneNormals[kek] = AlphaDir; }
				}
			}
			else {
				for (int kek = 0; kek < newPlaneNormals.size(); kek++) {
					auto position = find(minimumCut.begin(), minimumCut.end(), kek);
					if (position != minimumCut.end()) { newPlaneNormals[kek] = AlphaDir; }
					else { newPlaneNormals[kek] = BetaDir; }
				}
			}

			double EnergyValue_With_Hat = EnergyFunction(newPlaneNormals, numLinesDetected, LsdLinesVector, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, NumberOfPixelsInArea, PolygonIntersections);
			if (EnergyValue_With_Hat < InitialEnergyValue) {
				PlaneNormals = newPlaneNormals;
				InitialEnergyValue = EnergyValue_With_Hat;
				success = 1;
			}
		}
	}
}

void ShowLSDLinesOnScreen(cv::Mat image, vector<Line> LSDLines, int i) {
	int kk = 0;
	cv::Mat debi2 = image.clone();
	cv::cvtColor(debi2, debi2, CV_BGR2GRAY);
	cv::cvtColor(debi2, debi2, CV_GRAY2BGR);
	for (auto & l : LSDLines) {
		auto id = to_string(kk++);
		const cv::Scalar blaColor = cv::Scalar(255, 0, 0, 255);
		auto begin = cv::Point(l.begin.x, l.begin.y);
		auto end = cv::Point(l.end.x, l.end.y);
		cv::line(debi2, begin, end, blaColor, 2);
		//cv::circle(debi2, begin, 7, blaColor, -1);
		//cv::putText(debi2, "<" + id, begin, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, CV_RGB(0, 0, 0), 1.5);
		//cv::circle(debi2, end, 7, blaColor, -1);
		//cv::putText(debi2, ">", end, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, CV_RGB(0, 0, 0), 1.5);
	}
	if (i != -1) {
		cv::imshow("LSD Lines on a 1/" + std::to_string((int)pow(2, i + 1)) + " of a resolution", debi2);
	}
	else {
		cv::imshow("LSD Lines", debi2);
	}
	cv::waitKey(0);
	cv::destroyWindow("LSD_Lines");
}

void ShowLSDLinesOnScreen2(cv::Mat image, vector<Line> LSDLines, int i) {
	int kk = 0;
	cv::Mat debi2 = image.clone();
	cv::cvtColor(debi2, debi2, CV_BGR2GRAY);
	cv::cvtColor(debi2, debi2, CV_GRAY2BGR);
	for (auto & l : LSDLines) {
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
	if (i != -1) {
		cv::imshow("LSD Lines on a 1/" + std::to_string((int)pow(2, i + 1)) + " of a resolution", debi2);
	}
	else {
		cv::imshow("LSD Lines", debi2);
	}
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
	for (auto i : indices) {
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
	for (auto & p : lines) {
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
	//cv::waitKey(0);
	//cv::destroyWindow("AllJoinedLines");
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
	for (auto &l : LSDLines) {
		beggienings.push_back(l.begin);
	}



	auto b = cv::Mat(beggienings).reshape(1);
	cv::flann::Index btree(b, indexParams);


	std::vector<cv::Point3f> endings;
	endings.reserve(LSDLines.size());
	for (auto l : LSDLines) {
		endings.push_back(l.end);
	}

	auto e = cv::Mat(endings).reshape(1);
	cv::flann::Index etree(b, indexParams);

	// Iteration step
	for (auto & l : LSDLines) {
		const int size = 5;
		std::vector<float> dist(size);
		std::vector<int> indices(size);
		auto found = btree.radiusSearch(std::vector<float>{l.end.x, l.end.y, l.end.z}, indices, dist, 15, size, cv::flann::SearchParams(100));
		if (found < size) {
			indices.resize(found);
		}
		std::vector<Line> lines;
		lines.reserve(found);

		int j = 0;
		for (auto &line : LSDLines) {

			auto it = find_if(indices.begin(), indices.end(), [&](const int& i)
			{
				return i == line.index;
			});
			if (it != indices.end()) {
				lines.push_back(line);
			}
		}

		//TODO fix reorderting of one collection via values from another
		std::map<float, Line> m;
		for (auto & l : lines) {
			m[dist[j++]] = l;
		}
		int k = 0;
		for (auto & p : m) {
			lines[k++] = p.second;
		}

		if (lines.empty()) {
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


		if (l.index >= lineToJoin.index) { // only look behind
			l.index = lineToJoin.index;
		}
		else { // shall newer ocure!!!
			LSDLines[lineToJoin.index].index = l.index; //cant change LINQ copy 
		}

		///DEBUG_BEGIN
		//ShowLineAtIterationStep(image, indices, beggienings, endings, lineToJoin, l);
		///DEBUG_END
	}

	// correction step
	for (auto &l : LSDLines) {
		auto & line(output[l.index]);
		if (line.index == -1) {
			line = l;
		}
		else { // Todo check vertical line grouth!
			if (line.begin.x > l.begin.x) {
				line.begin = l.begin;
			}

			if (line.end.x < l.end.x) {
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
	for (auto &l : LSDLines) {
		beggienings.push_back(l.begin);
	}



	auto b = cv::Mat(beggienings).reshape(1);
	cv::flann::Index btree(b, indexParams);


	std::vector<cv::Point3f> endings;
	endings.reserve(LSDLines.size());
	for (auto l : LSDLines) {
		endings.push_back(l.end);
	}

	auto e = cv::Mat(endings).reshape(1);
	cv::flann::Index etree(e, indexParams);

	// Iteration step
	for (auto & l : LSDLines) {
		const int size = 5;
		std::vector<float> dist(size);
		std::vector<int> indices(size);
		auto found = etree.radiusSearch(std::vector<float>{l.begin.x, l.begin.y, l.begin.z}, indices, dist, 15, size, cv::flann::SearchParams(100));
		if (found < size) {
			indices.resize(found);
		}
		std::vector<Line> lines;
		lines.reserve(found);

		int j = 0;
		for (auto &line : LSDLines) {

			auto it = find_if(indices.begin(), indices.end(), [&](const int& i) {
				return i == line.index;
			});
			if (it != indices.end()) {
				lines.push_back(line);
			}
		}

		//TODO fix reorderting of one collection via values from another
		std::map<float, Line> m;
		for (auto & l : lines) {
			m[dist[j++]] = l;
		}
		int k = 0;
		for (auto & p : m) {
			lines[k++] = p.second;
		}

		if (lines.empty()) {
			continue;
		}


		Line lineToJoin;
		auto it = find_if(lines.begin(), lines.end(), [&](const Line & r) {
			return l.end.z == r.end.z;
		});
		if (it != lines.end()) {
			lineToJoin = *it;
		}
		else {
			continue;
		}


		if (l.index >= lineToJoin.index) { // only look behind
			l.index = lineToJoin.index;
		}
		else { // shall newer ocure!!!
			LSDLines[lineToJoin.index].index = l.index; //cant change LINQ copy 
		}

		///DEBUG_BEGIN
		//ShowLineAtIterationStep(image, indices, beggienings, endings, lineToJoin, l);
		///DEBUG_END
	}

	// correction step
	for (auto &l : LSDLines) {
		auto & line(output[l.index]);
		if (line.index == -1) {
			line = l;
		}
		else { // Todo check vertical line grouth!
			if (line.begin.x > l.begin.x) {
				line.begin = l.begin;
			}

			if (line.end.x < l.end.x) {
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

void getLinePairs(vector <cv::Point3f> lines, vector<PairOfTwoLines>& outputVec) {
	double EPSILON = 0.000001;

	for (int i = 0; i < lines.size() / 2; i++) {
		Line firstLine = Line(lines[2 * i], lines[2 * i + 1]);
		for (int j = 0; j < lines.size() / 2; j++) {
			Line secondLine = Line(lines[2 * j], lines[2 * j + 1]);
			if (doLinesIntersect(firstLine, secondLine, EPSILON)) {
				PairOfTwoLines tempPair = PairOfTwoLines(i, j);
				outputVec.push_back(tempPair);
			}
		}
	}
}
///INTERSECTION_DETECTION_END

void erosion(std::string src, cv::Mat &result) {
	auto M = cv::imread(src, CV_LOAD_IMAGE_GRAYSCALE);
	auto Orig = cv::imread(src, CV_LOAD_IMAGE_COLOR);

	resize(M, M, cv::Size(640, 480));
	resize(Orig, Orig, cv::Size(640, 480));
	cv::imshow("Original Image", Orig);
	cv::medianBlur(M, M, 37);
	//equalizeHist(M, M); // повышение контрастности

	auto erosion_type = cv::MORPH_ERODE;
	auto erosion_size = 3;
	auto element = cv::getStructuringElement(erosion_type,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	erode(M, M, element); // увеличение черноты

	auto clahe = cv::createCLAHE(50, cv::Size(16, 16));
	clahe->apply(M, M);
	erode(M, M, element); // повышение контрастности HDR style

	bitwise_not(M, M); // инверсия
	result = M.clone();
}

void getSkeleton(cv::Mat img, cv::Mat& result) {
	cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);
	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do {
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);

		done = (cv::countNonZero(img) == 0);
	} while (!done);
	result = skel.clone();
}

void getLsdLines(cv::Mat image, vector<Line> &LsdLines, int &numLinesDetected, vector<cv::Point3f> &outputLsdLinesVector) {
	//LSD
	double* LsdLinesArray = DoLSD(image, numLinesDetected);
	cout << "Number of LSD lines detected: " << numLinesDetected << endl;

	//Копируем точки из массива в вектор с учетом порядка
	vector<cv::Point3f> LsdLinesVector; //элемент с номером 2*i дает нам точку начала i-ой линии, элемент с номером 2*i + 1 дает нам точку конца i-ой линии
	for (int i = 0; i < numLinesDetected; i++) {
		double x1 = RoundTo(LsdLinesArray[7 * i]);
		double y1 = RoundTo(LsdLinesArray[7 * i + 1]);
		double x2 = RoundTo(LsdLinesArray[7 * i + 2]);
		double y2 = RoundTo(LsdLinesArray[7 * i + 3]);
		cv::Point3f point_one = cv::Point3f(x1, y1, 0);
		cv::Point3f point_two = cv::Point3f(x2, y2, 0);
		if (x1 < x2) { // todo use epsilon!
			LsdLinesVector.push_back(point_one);
			LsdLinesVector.push_back(point_two);
		}
		else if (x2 < x1) {
			LsdLinesVector.push_back(point_two);
			LsdLinesVector.push_back(point_one);
		}
		else if (y1 < y2) {
			LsdLinesVector.push_back(point_one);
			LsdLinesVector.push_back(point_two);
		}
		else {
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

	outputLsdLinesVector = LsdLinesVector;

	//todo: split into ~3 line long seprate member functions
	// Sorted left->right lines layer
	vector<Line> LSDLines = [](auto LsdLines) //todo extract method
	{
		vector<Line> result(LsdLines.size() / 2);
		for (int i = 0; i < result.size(); ++i) {
			result[i] = Line(LsdLines[i * 2], LsdLines[i * 2 + 1], i);
		}
		return result;
	}(LsdLinesVector);
	LsdLines = LSDLines;
}

class IntersectionPoint {
public:
	int otherIndex;
	cv::Point3f interPoint;

	double distance;

	IntersectionPoint(int b, cv::Point3f c, double d) {
		otherIndex = b;
		interPoint = c;
		distance = d;
	}

};

class myPair {
public:
	cv::Point3f interPoint;
	int index;
	double distance;

	myPair(cv::Point3f a, double b, int c) {
		interPoint = a;
		distance = b;
		index = c;
	}
};


int main() {

	clock_t tStart = clock();
	srand(time(0)); //чтобы последовательность рандомных чисел была всегда уникальна при новом запуске

					//Переменные для настройки
	double focal_length = 4; //фокальное расстояние в mm
	double sensor_width = 4.59; //ширина сенсора в mm
	double ExtendThreshold = 0.01; //порог отклонения линии (для удлинения)
	double countInlierThreshold = 0.0001; //если квадрат скалярного произведения двух линий меньше этого числа, то мы считаем эти линии ортогональными
	double noiseFilterConst = 10;

	double AngleTolerance = 0.01; //если abs(tg(angle1) - tg(angle2)) < AngleTolerance, то эти две линии объединяются в одну группу (параллельных линий с некоторой степенью толерантности)
	double distanceEpsilon = 0;

	double AngleEpsilon = 0.55;
	double step = 2; //шаг для дискритизации линий
	double radius = 30; //радиус поиска около точек соединительных линий

	uint ResizeIfMoreThan = 900; //если ширина или высота изображения больше этого числа, то мы меняем размер изображения
	bool debug = 1;
	bool useSkeleton = 0;
	bool useLinesExt = 1;

	//Открытие изображения
	auto src = "usecase_2.png";
	cv::Mat image;
	if (!useSkeleton) {
		image = cv::imread(src, CV_LOAD_IMAGE_COLOR);
	}
	else {
		cv::Mat erodedImage;
		erosion(src, erodedImage); //эрозия
		cv::imshow("Eroded image", erodedImage);
		BrightnessAndContrastAuto(erodedImage, erodedImage, 75);
		cv::imshow("Eroded image Contrast", erodedImage);
		cv::Mat skeleton;
		getSkeleton(erodedImage, skeleton); //получение скелета
		cv::imshow("skeleton", skeleton);
		cv::imwrite("skeleton.jpg", skeleton);

		image = cv::imread("skeleton.jpg", CV_LOAD_IMAGE_COLOR);
	}


	//Изменение размера
	uint maxRes = max(image.cols, image.rows);
	if (maxRes > ResizeIfMoreThan) {
		float scaleFactor = float(ResizeIfMoreThan) / maxRes;
		cv::Size size = cv::Size(image.cols * scaleFactor, image.rows * scaleFactor);
		cv::resize(image, image, size, 0, 0, CV_INTER_CUBIC);
	}
	//BrightnessAndContrastAuto(image, image, 25);


	/*
	auto t2 = chrono::high_resolution_clock::now().time_since_epoch();
	cv::imwrite("not_sharpened.jpg",image);
	cv::Mat frame = image.clone();
	cv::Mat output;
	cv::GaussianBlur(frame, output, cv::Size(17, 23), 3);
	cv::addWeighted(frame, 1.50, output, -0.5, 0, output);
	cv::imwrite("sharpened.jpg", output);
	auto dt2 = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch() - t2).count();
	cout << "it took " << dt2 << endl; // 247ms for 10 iterations, 10ms for one iteration, 35ms for 2 on one core


	//B
	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;
	cv::Mat src_gray;
	cv::cvtColor(image, src_gray, CV_BGR2GRAY);
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
	/// Gradient Y
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	cv::Mat grad;
	addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, grad);
	cv::imshow("sobel_x", abs_grad_x);
	cv::imwrite("sobel_x.jpg", abs_grad_x);
	cv::imshow("sobel_y", abs_grad_y);
	cv::imwrite("sobel_y.jpg", abs_grad_y);
	cv::imshow("sobel", grad);
	cv::imwrite("sobel_summ.jpg", grad);
	cv::waitKey();
	//B*/


	//Границы кадра
	uint maxX = image.cols;
	uint maxY = image.rows;

	//Фокальное расстояние
	double temp_focal = maxX * focal_length / sensor_width;
	uint f = (uint)temp_focal; //фокальное расстояние в пикселях

	int numLinesDetected;
	vector<cv::Point3f> LsdLinesVector;
	vector<Line> LSDLines;
	vector<cv::Point3f> ExtendedLinesVector;

	if (!useLinesExt)
	{
		//LSD
		getLsdLines(image, LSDLines, numLinesDetected, LsdLinesVector);

		///DEBUG_BEGIN
		ShowLSDLinesOnScreen(image, LSDLines, -1);
		///DEBUG_END

		/*
		auto t2 = chrono::high_resolution_clock::now().time_since_epoch();
		///PYRAMID BEGIN
		cv::Mat copiedImage = image.clone();
		for (int i = 0; i < 3; i++) {
		pyrDown(copiedImage, copiedImage, cv::Size(copiedImage.cols / 2, copiedImage.rows / 2));
		//cv::imshow("1/" + std::to_string((int)pow(2,i+1)) + " of a resolution", copiedImage);

		int num;
		vector<cv::Point3f> LsdLinesVec;
		vector<Line> LsdLines;
		getLsdLines(copiedImage, LsdLines, num, LsdLinesVec);
		ShowLSDLinesOnScreen(copiedImage, LsdLines, i);
		}
		///PYRAMID END
		auto dt2 = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch() - t2).count();
		cout << "it took " << dt2 << endl; // 247ms for 10 iterations, 10ms for one iteration, 35ms for 2 on one core*/



		//A
		vector<cv::Mat> imageStorage;
		vector<vector<cv::Point3f>> LinesVecStorage;
		vector<vector<Line>> LsdStorage;
		cv::Mat copiedImage = image.clone();

		for (int i = 0; i < 3; i++) {
			pyrDown(copiedImage, copiedImage, cv::Size(copiedImage.cols / 2, copiedImage.rows / 2));
			int num;
			vector<cv::Point3f> LsdLinesVec;
			vector<Line> LsdLines;
			getLsdLines(copiedImage, LsdLines, num, LsdLinesVec);

			imageStorage.push_back(copiedImage);
			LinesVecStorage.push_back(LsdLinesVec);
			LsdStorage.push_back(LsdLines);
			//ShowLSDLinesOnScreen(copiedImage, LsdLines, i);
		}

		cv::Mat component;
		for (int i = 2; i >= 0; i--) {
			cv::Mat currentIm = imageStorage[i].clone();
			vector<Line> LSDLines = LsdStorage[i];

			for (auto & l : LSDLines) {
				cv::Scalar blaColor;
				if (i == 2) {
					blaColor = cv::Scalar(255, 0, 0, 255);
				}
				else if (i == 1) {
					blaColor = cv::Scalar(0, 255, 0, 255);
				}
				else {
					blaColor = cv::Scalar(0, 0, 255, 255);
				}
				auto begin = cv::Point(l.begin.x, l.begin.y);
				auto end = cv::Point(l.end.x, l.end.y);
				cv::line(currentIm, begin, end, blaColor, 3);
			}

			if (i != 2) {
				for (int j = 0; j < component.rows; j++)
				{
					for (int k = 0; k < component.cols; k++)
					{
						cv::Vec3b bgrPixel = component.at<cv::Vec3b>(j, k);
						if (i == 1) {
							if (bgrPixel.val[0] == 255) {
								currentIm.at<cv::Vec3b>(j, k) = bgrPixel;
							}
						}
						else if (i == 0)
						{
							if (bgrPixel.val[0] == 255 || bgrPixel.val[1] == 255) {
								currentIm.at<cv::Vec3b>(j, k) = bgrPixel;
							}
						}

					}
				}
			}
			imageStorage[i] = currentIm.clone();
			pyrUp(currentIm, currentIm, cv::Size(currentIm.cols * 2, currentIm.rows * 2));
			component = currentIm.clone();
		}
		//A

		//A
		cv::Mat currentIm = image.clone();
		for (auto & l : LSDLines) {
			const cv::Scalar blaColor = cv::Scalar(255, 255, 0, 255);
			auto begin = cv::Point(l.begin.x, l.begin.y);
			auto end = cv::Point(l.end.x, l.end.y);
			cv::line(currentIm, begin, end, blaColor, 3);
		}
		for (int j = 0; j < component.rows; j++)
		{
			for (int k = 0; k < component.cols; k++)
			{
				cv::Vec3b bgrPixel = component.at<cv::Vec3b>(j, k);
				if (bgrPixel.val[0] == 255 || bgrPixel.val[1] == 255 || bgrPixel.val[2] == 255) {
					currentIm.at<cv::Vec3b>(j, k) = bgrPixel;
				}
			}
		}

		for (int i = 2; i >= 0; i--) {
			cv::imshow(std::to_string(i), imageStorage[i]);
			cv::waitKey();
		}
		cv::imshow(std::to_string(-1), currentIm);
		cv::waitKey();
		//A





		unordered_map<int, Line> lines;

		auto t1 = chrono::high_resolution_clock::now().time_since_epoch();
		for (int i = 0; i < 2; ++i) {
			lines = unordered_map<int, Line>();
			ExtendLines(image, LSDLines, lines);
			//ExtendLinesReversalMove(image, LSDLines, lines); //обратный ход
			LSDLines = vector<Line>();
			LSDLines.resize(lines.size());
			cout << "reduced on iteration " << i << " lines count: " << lines.size() << endl;
			for (auto &p : lines) {
				//noise filter
				if (p.second.norm() >(noiseFilterConst*maxX / 100)) {
					LSDLines.push_back(p.second);
				}
			}
		}
		auto dt = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch() - t1).count();
		cout << "it took " << dt << endl; // 247ms for 10 iterations, 10ms for one iteration, 35ms for 2 on one core

										  ///DEBUG_BEGIN
		ShowJoinedLines(image, lines);
		///DEBUG_END

		for (auto &p : LSDLines) {
			Line temp = p;
			ExtendedLinesVector.push_back(temp.begin);
			ExtendedLinesVector.push_back(temp.end);
		}
	}

	else {
		getLsdLines(image, LSDLines, numLinesDetected, LsdLinesVector);

		/*
		Line mainLine = Line(cv::Point3f(39, 123, 0), cv::Point3f(174, 124, 0), 0);
		LSDLines.push_back(mainLine);*/

		Line leftVerticalLine = Line(cv::Point3f(0, 0, 0), cv::Point3f(0, maxY, 0), 10000);
		Line rightVerticalLine = Line(cv::Point3f(maxX, 0, 0), cv::Point3f(maxX, maxY, 0), 20000);
		Line upHorizontalLine = Line(cv::Point3f(0, 0, 0), cv::Point3f(maxX, 0, 0), 30000);
		Line downHorizontalLine = Line(cv::Point3f(0, maxY, 0), cv::Point3f(maxX, maxY, 0), 40000);

		LSDLines.push_back(leftVerticalLine);
		LSDLines.push_back(rightVerticalLine);
		LSDLines.push_back(upHorizontalLine);
		LSDLines.push_back(downHorizontalLine);




		vector<IntersectionPoint> interBeginStorage;
		vector<IntersectionPoint> interEndStorage;

		for (int i = 0; i < LSDLines.size(); i++) {
			Line currentLine = LSDLines[i];
			double A1 = currentLine.begin.y - currentLine.end.y;
			double B1 = currentLine.end.x - currentLine.begin.x;
			double C1 = currentLine.begin.x*currentLine.end.y - currentLine.end.x*currentLine.begin.y;
			vector<myPair> currentBeginPoints;
			vector<myPair> currentEndPoints;


			for (int j = 0; j < LSDLines.size(); j++) {
				if (j != i) {
					Line other = LSDLines[j];
					double A2 = other.begin.y - other.end.y;
					double B2 = other.end.x - other.begin.x;
					double C2 = other.begin.x*other.end.y - other.end.x*other.begin.y;
					if ((A1*B2 - A2*B1) != 0) { //если линии не параллельны
						double x = -(C1*B2 - C2*B1) / (A1*B2 - A2*B1);
						double y = -(A1*C2 - A2*C1) / (A1*B2 - A2*B1);

						if (x >= other.begin.x && x <= other.end.x && y >= other.begin.y && y <= other.end.y) {
							double begDist = sqrt((currentLine.begin.x - x)*(currentLine.begin.x - x) + (currentLine.begin.y - y)*(currentLine.begin.y - y));
							double endDist = sqrt((currentLine.end.x - x)*(currentLine.end.x - x) + (currentLine.end.y - y)*(currentLine.end.y - y));
							if (begDist < endDist) {
								myPair pair = myPair(cv::Point3f(x, y, 0), begDist, j);
								currentBeginPoints.push_back(pair);
							}
							else {
								myPair pair = myPair(cv::Point3f(x, y, 0), endDist, j);
								currentEndPoints.push_back(pair);
							}
						}

					}
				}
			}

			sort(currentBeginPoints.begin(), currentBeginPoints.end(), [&](myPair a, myPair b) {
				return a.distance < b.distance;
			});
			sort(currentEndPoints.begin(), currentEndPoints.end(), [&](myPair a, myPair b) {
				return a.distance < b.distance;
			});

			if (currentBeginPoints.size() > 0) {
				IntersectionPoint begin = IntersectionPoint(currentBeginPoints[0].index, currentBeginPoints[0].interPoint, currentBeginPoints[0].distance);
				interBeginStorage.push_back(begin);
			}
			else {
				IntersectionPoint badBeg = IntersectionPoint(-1, cv::Point3f(-1, -1, -1), -1);
				interBeginStorage.push_back(badBeg);
			}
			if (currentEndPoints.size() > 0) {
				IntersectionPoint end = IntersectionPoint(currentEndPoints[0].index, currentEndPoints[0].interPoint, currentEndPoints[0].distance);
				interEndStorage.push_back(end);
			}
			else {
				IntersectionPoint badEnd = IntersectionPoint(-1, cv::Point3f(-1, -1, -1), -1);
				interEndStorage.push_back(badEnd);
			}
		}

		ShowLSDLinesOnScreen(image, LSDLines, -1);

		double extension_fac = 0.01;
		double thresh = 15;
		//продление
		for (int i = 0; i < LSDLines.size(); i++) {
			Line currentLine = LSDLines[i];

			double x1 = currentLine.begin.x;
			double y1 = currentLine.begin.y;
			double x2 = currentLine.end.x;
			double y2 = currentLine.end.y;

			double line_length = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
			double line_gradient, rise_angle;
			if (x2 != x1) {
				line_gradient = (y2 - y1) / (x2 - x1);
				rise_angle = atan(abs(line_gradient));
			}
			else {
				line_gradient = 999999;
				rise_angle = M_PI / 2;
			}

			double lenExt = line_length*extension_fac;

			if (interBeginStorage[i].otherIndex != -1) {
				int count = floor(interBeginStorage[i].distance / lenExt);
				for (int j = 0; j < count; j++) {
					double delta_x = cos(rise_angle)*(lenExt);
					double delta_y = sin(rise_angle)*(lenExt);
					double x_temp, y_temp;
					if (line_gradient == 999999) {
						y_temp = y1 - delta_y;
						x_temp = x1;
					}
					else if (line_gradient > 0) {
						x_temp = x1 - delta_x;
						y_temp = y1 - delta_y;
					}
					else {
						x_temp = x1 - delta_x;
						y_temp = y1 + delta_y;
					}

					int xx = RoundTo(x_temp);
					if (xx == maxX) xx = maxX - 1;
					int yy = RoundTo(y_temp);
					if (yy == maxY) yy = maxY - 1;

					int xxx = RoundTo(x1);
					if (xxx == maxX) xxx = maxX - 1;
					int yyy = RoundTo(y1);
					if (yyy == maxY) yyy = maxY - 1;
					cv::Vec3b currentColor = image.at<cv::Vec3b>(yyy, xxx);
					cv::Vec3b mColor = image.at<cv::Vec3b>(yy, xx);
					double dist = sqrt((currentColor.val[0] - mColor.val[0])*(currentColor.val[0] - mColor.val[0]) + (currentColor.val[1] - mColor.val[1])*(currentColor.val[1] - mColor.val[1]) + (currentColor.val[2] - mColor.val[2])*(currentColor.val[2] - mColor.val[2]));
					if (dist <= thresh) {
						x1 = x_temp;
						y1 = y_temp;
					}
					else {
						break;
					}
				}

				double dist = sqrt((x1 - interBeginStorage[i].interPoint.x)*(x1 - interBeginStorage[i].interPoint.x) + (y1 - interBeginStorage[i].interPoint.y)*(y1 - interBeginStorage[i].interPoint.y));
				if (dist <= lenExt) {
					x1 = interBeginStorage[i].interPoint.x;
					y1 = interBeginStorage[i].interPoint.y;
				}
			}

			if (interEndStorage[i].otherIndex != -1) {
				int count = floor(interEndStorage[i].distance / lenExt);
				for (int j = 0; j < count; j++) {
					double delta_x = cos(rise_angle)*(lenExt);
					double delta_y = sin(rise_angle)*(lenExt);
					double x_temp, y_temp;
					if (line_gradient == 999999) {
						y_temp = y2 + delta_y;
						x_temp = x2;
					}
					else if (line_gradient > 0) {
						x_temp = x2 + delta_x;
						y_temp = y2 + delta_y;
					}
					else {
						x_temp = x2 + delta_x;
						y_temp = y2 - delta_y;
					}

					int xx = RoundTo(x_temp);
					if (xx == maxX) xx = maxX - 1;
					int yy = RoundTo(y_temp);
					if (yy == maxY) yy = maxY - 1;

					int xxx = RoundTo(x2);
					if (xxx == maxX) xxx = maxX - 1;
					int yyy = RoundTo(y2);
					if (yyy == maxY) yyy = maxY - 1;
					cv::Vec3b currentColor = image.at<cv::Vec3b>(yyy, xxx);
					cv::Vec3b mColor = image.at<cv::Vec3b>(yy, xx);

					double dist = sqrt((currentColor.val[0] - mColor.val[0])*(currentColor.val[0] - mColor.val[0]) + (currentColor.val[1] - mColor.val[1])*(currentColor.val[1] - mColor.val[1]) + (currentColor.val[2] - mColor.val[2])*(currentColor.val[2] - mColor.val[2]));
					if (dist <= thresh) {
						x2 = x_temp;
						y2 = y_temp;
					}
					else {
						break;
					}
				}

				double dist = sqrt((x2 - interEndStorage[i].interPoint.x)*(x2 - interEndStorage[i].interPoint.x) + (y2 - interEndStorage[i].interPoint.y)*(y2 - interEndStorage[i].interPoint.y));
				if (dist <= lenExt) {
					x2 = interEndStorage[i].interPoint.x;
					y2 = interEndStorage[i].interPoint.y;
				}
			}

			currentLine.begin.x = x1;
			currentLine.begin.y = y1;
			currentLine.end.x = x2;
			currentLine.end.y = y2;

			LSDLines[i] = currentLine;
		}


		ShowLSDLinesOnScreen(image, LSDLines, -1);
		vector<Line> atata;
		for (int k = 0; k < LSDLines.size() - 4; k++) {
			Line temp = LSDLines[k];
			if (temp.norm() >= noiseFilterConst*maxX / 100) {
				atata.push_back(temp);
				ExtendedLinesVector.push_back(temp.begin);
				ExtendedLinesVector.push_back(temp.end);
			}

		}
		ShowLSDLinesOnScreen(image, atata, -1);
		
	}

	//Getting intersections
	vector<PairOfTwoLines> LinePairsVector;
	getLinePairs(ExtendedLinesVector, LinePairsVector);
	cout << "Number of line pairs: " << LinePairsVector.size() << endl;

	numLinesDetected = ExtendedLinesVector.size() / 2;

	//RANSAC (находим углы альфа и бэта)
	uint maxRansacTrials = (uint)LinePairsVector.size(); //максимальное количество итерация алгоритма RANSAC
	dlib::matrix<double, 0, 1> solution = RANSAC_(maxRansacTrials, countInlierThreshold, f, LinePairsVector, ExtendedLinesVector);
	cout << "angle Alpha (in radians): " << solution(0) << endl << "angle Beta (in radians): " << solution(1) << endl;

	//Получаем vanishing points
	vector<cv::Point3f> VanishingPoints;
	getVanishingPoints(solution(0), solution(1), f, VanishingPoints);

	cv::Mat debi2 = image.clone();

	const cv::Scalar blaColor1 = cv::Scalar(0, 0, 255, 255);
	const cv::Scalar blaColor2 = cv::Scalar(0, 255, 0, 255);
	const cv::Scalar blaColor3 = cv::Scalar(255, 0, 0, 255);

	auto a = cv::Point(VanishingPoints[0].x, VanishingPoints[0].y);
	auto b = cv::Point(VanishingPoints[1].x, VanishingPoints[1].y);
	auto c = cv::Point(VanishingPoints[2].x, VanishingPoints[2].y);

	cv::circle(debi2, a, 15, blaColor1, -1);
	cv::circle(debi2, b, 15, blaColor2, -1);
	cv::circle(debi2, c, 15, blaColor3, -1);

	cv::imshow("aaa", debi2);
	cv::waitKey();

	//Находим направление каждой линии
	vector<uint> DirectionsOfLines; //элемент с номером i означает направление i-ой линии (0=x, 1=y, 2=z, 3=xy, 4=xz, 5=yz)
	assignDirections(numLinesDetected, ExtendedLinesVector, VanishingPoints, DirectionsOfLines);

	//return 0;


	/// Past
	for (int i = 0; i < ExtendedLinesVector.size(); i++) {
		ExtendedLinesVector[i].z = 0;
	}
	//Выделяем все замкнутые области
	vector<vector<uint> > PolygonsVector;
	getPolygons(numLinesDetected, ExtendedLinesVector, AngleTolerance, step, radius, PolygonsVector, true, image, distanceEpsilon, AngleEpsilon);


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

	using namespace cv;

	cv::Mat overlay = image.clone();


	//Получаем глубины
	cv::Mat Depths;
	getDepths(focal_length, PlaneNormals, numLinesDetected, ExtendedLinesVector, DirectionsOfLines, PolygonsVector, PolygonIntersections, Depths);
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
		vector<cv::Point> convexHull;  // Convex hull points 
		vector<cv::Point> contour;  // Convex hull contour points        
		double epsilon = 0.001; // Contour approximation accuracy

								// Calculate convex hull of original points (which points positioned on the boundary)
		cv::convexHull(cv::Mat(tempVec), convexHull, false);
		approxPolyDP(cv::Mat(convexHull), contour, 0.001, true);
		cv::Scalar blaColor;
		int val1 = rand() % 255;
		int val2 = rand() % 255;
		int val3 = rand() % 255;

		if (PlaneNormals[i] == 0) {
			blaColor = cv::Scalar(255, val2-50, 0);
		}
		else if (PlaneNormals[i] == 1) {
			blaColor = cv::Scalar(0, 255, val3-50);
		}
		else {
			blaColor = cv::Scalar(val1-50, 0, 255);
		}
		cv::fillConvexPoly(overlay, &contour[0], contour.size(), blaColor);
		double alpha = 0.3;
		cv::addWeighted(image, alpha, overlay, 1 - alpha, 0, overlay);

	}
	cv::imshow("PolygonsTransparent", overlay);
	cv::waitKey(0);
	cv::destroyWindow("Polygons");

	system("pause");
	return 0;
}