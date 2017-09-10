#include <chrono>
#include <tools.h>
typedef uint32_t uint;
using namespace std;

NearestLine::NearestLine(int a, double b) {
	index = a;
	dist = b;
}

LineScore::LineScore(uint a, uint b, uint c) {
	goodPoints = a;
	LineIndex = c;
	totalPoints = b;
}

PairOfTwoLines::PairOfTwoLines(uint a, uint b) {
	FirstIndex = a;
	SecondIndex = b;
}

PairOfTwoLines::PairOfTwoLines() {
	FirstIndex = 0;
	SecondIndex = 0;
}

//Для сортировки
bool comparator(const LineScore l, const LineScore r) {
	double score1 = 0;
	double score2 = 0;
	if (l.totalPoints != 0) { score1 = double(l.goodPoints) / l.totalPoints; }
	if (r.totalPoints != 0) { score2 = double(r.goodPoints) / r.totalPoints; }
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