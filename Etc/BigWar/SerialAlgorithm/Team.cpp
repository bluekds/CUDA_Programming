#include "Team.h"

Team::Team(UINT _numUnit, FILE *_fp)
{
	armies = NULL;
	numArmies = _numUnit;
	if (numArmies > 0)
		armies = new Army[numArmies];
	fp = _fp;
}

Team::~Team()
{
	if (armies != NULL)
		delete armies;
}

int Team::loadTeam()
{
	int ID = 0;
	for (ID = 0; ID < numArmies; ID++) {
		if (fread_s(armies[ID].pos, sizeof(float) * 3, sizeof(float), 3, fp) == 0) {
			numArmies = ID;
			break;
		}
	}

	return numArmies;
}
