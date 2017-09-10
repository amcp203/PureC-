#include <opencv2/core/core.hpp>
#include <vector>
#include <rectifying_homography.h>

typedef uint32_t uint;
using namespace std;


void solveSystem(cv::Mat input, cv::Mat &output) {
	cv::Mat S, U, Vt;
	cv::SVD::compute(input, S, U, Vt, cv::SVD::FULL_UV);

	cv::Mat V;
	cv::transpose(Vt, V);

	int lastColumn = V.cols - 1;
	cv::Mat lastColumnMatrix = V.col(lastColumn).clone();
	output = lastColumnMatrix.clone();
}

void getDepths(double focal, vector<uint>& PlaneNormals, int numLinesDetected, vector<cv::Point3f> ExtendedLinesVector, vector<uint> DirectionsOfLines, vector<vector<uint> > PolygonsVector, bool** PolygonIntersections, cv::Mat &Depths) {

	vector<vector<double>> input;

	for (int i = 0; i < PolygonsVector.size() - 1; i++) {
		for (int j = i + 1; j < PolygonsVector.size(); j++) {
			if (PolygonIntersections[i][j] == 1) {
				//Получение вектора нормали i-ого сегмента
				int normal_i = PlaneNormals[i];
				cv::Mat norm_i;
				switch (normal_i)
				{
				case 0:
					norm_i = (cv::Mat_<double>(1, 3) << 1, 0, 0);
					break;
				case 1:
					norm_i = (cv::Mat_<double>(1, 3) << 0, 1, 0);
					break;
				case 2:
					norm_i = (cv::Mat_<double>(1, 3) << 0, 0, 1);
					break;
				}

				//Получение вектора нормали j-ого сегмента
				int normal_j = PlaneNormals[j];
				cv::Mat norm_j;
				switch (normal_j)
				{
				case 0:
					norm_j = (cv::Mat_<double>(1, 3) << 1, 0, 0);
					break;
				case 1:
					norm_j = (cv::Mat_<double>(1, 3) << 0, 1, 0);
					break;
				case 2:
					norm_j = (cv::Mat_<double>(1, 3) << 0, 0, 1);
					break;
				}

				//Получение вектора общей линии (пересечения)
				cv::Mat K_inverted = K_inv(focal);
				cv::Mat commonLine;

				vector<uint> planeA = PolygonsVector[i];
				vector<uint> planeB = PolygonsVector[j];
				for (int k1 = 0; k1 < planeA.size(); k1++) {
					for (int k2 = 0; k2 < planeB.size(); k2++) {
						if (planeA[k1] == planeB[k2]) {
							//Преобразование point3f в mat
							cv::Point3f commonPoint = ExtendedLinesVector[2 * planeA[k1] + 1];
							double temporaryArray[3][1] = { { commonPoint.x },{ commonPoint.y },{ commonPoint.z } };
							cv::Mat commonPointMatrix = cv::Mat(3, 1, CV_64F, temporaryArray);
							commonLine = K_inverted*commonPointMatrix;
						}
					}
				}


				vector<double> row(PolygonsVector.size(), 0); //создание вектора, заполненного нулями
				cv::Mat OneElementMat_1 = norm_j*commonLine;
				row[i] = OneElementMat_1.at<double>(0, 0);
				cv::Mat OneElementMat_2 = -1 * norm_i*commonLine;
				row[j] = OneElementMat_2.at<double>(0, 0);

				input.push_back(row);
			}
		}
	}
	//Преобразуем вектор векторов в двумерный массив
	int verticalSize = input.size();
	if (verticalSize != 0) {
		int horizontalSize = input[0].size();
		double **inputArray = new double *[verticalSize];;
		for (int j = 0; j < verticalSize; j++) {
			inputArray[j] = new double[horizontalSize];
		}
		for (int i = 0; i < verticalSize; i++) {
			for (int j = 0; j < horizontalSize; j++) {
				inputArray[i][j] = input[i][j];
			}
		}
		cv::Mat inputMat = cv::Mat(verticalSize, horizontalSize, CV_32F, inputArray);
		solveSystem(inputMat, Depths);
	}
}
