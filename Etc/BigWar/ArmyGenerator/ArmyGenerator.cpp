#include "ArmyGenerator.h"
#include <time.h>

ArmyGenerator::ArmyGenerator(UINT _min, UINT _max)
{
	range[0] = _min;
	range[1] = _max;
	gen = new std::mt19937_64(rd());
	dist = new std::uniform_int_distribution<UINT>(range[0], range[1]);
	srand(time(NULL));
}

ArmyGenerator::~ArmyGenerator()
{
	delete dist;
	delete gen;
}

Army ArmyGenerator::genArmy()
{
	for (int i = 0; i < ARMY_DIMENSION; i++)
		army.pos[i] = (*dist)(*gen) + (rand() % 100) / 100.f;
	return army;
}
