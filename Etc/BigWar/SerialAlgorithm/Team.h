#pragma once
#include "../Common/BigWar.h"

class Team
{
public:
	Team(UINT _numUnit, FILE* _fp);
	~Team();
	int loadTeam();

public:
	UINT numArmies;
	Army* armies;
	
private:
	FILE* fp;
};

