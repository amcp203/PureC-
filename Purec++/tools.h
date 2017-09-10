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

//��� ����������
bool comparator(const LineScore l, const LineScore r);

//��������� ���. � ��������� �� a �� b
float RandomFloat(float a, float b);

//��������� ������������� ����� � ��������� �� a �� b
uint RandomInt(uint a, uint b);

//���������� �� ������ �� �������� ����������
double RoundTo(double x);