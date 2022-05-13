#pragma once
#include <random>
#include "../Common/BigWar.h"

class ArmyGenerator
{
public:
	ArmyGenerator(UINT _min, UINT _max);
	~ArmyGenerator();
	Army genArmy();

private:
	Army army;
	UINT range[2];
	std::random_device rd;
	std::mt19937_64* gen;
	std::uniform_int_distribution<UINT>* dist;
};

