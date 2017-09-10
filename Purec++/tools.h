#pragma once
#include <chrono>
typedef uint32_t uint;

class NearestLine {
public:
	int index;
	double dist;
	NearestLine(int a, double b);
};

class LineScore {
public:
	uint goodPoints;
	uint totalPoints;
	uint LineIndex;
	LineScore(uint a, uint b, uint c);
};

class PairOfTwoLines {
public:
	uint FirstIndex;
	uint SecondIndex;

	PairOfTwoLines(uint a, uint b);

	PairOfTwoLines();
};

//Для сортировки
bool comparator(const LineScore l, const LineScore r);

//Рандомное вещ. в интервале от a до b
float RandomFloat(float a, float b);

//Рандомное положительное целое в интервале от a до b
uint RandomInt(uint a, uint b);

//Округление до целого по правилам округления
double RoundTo(double x);