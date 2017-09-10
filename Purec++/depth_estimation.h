#pragma once
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;

void solveSystem(cv::Mat input, cv::Mat &output);

void getDepths(double focal, vector<uint>& PlaneNormals, int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, bool** PolygonIntersections, cv::Mat &Depths);
