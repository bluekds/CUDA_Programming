#include "../Common/BigWar.h"
#include "Team.h"

int main(int argc, char** argv)
{
	if (argc < 4) {
		printf("Usage: exe inputA inputB outputFile\n");
		exit(1);
	}

	FILE* fp = NULL;
	UINT numArmies = 0;	
	Team *teams[2];

	for (int i = 0; i < 2; i++) {
		fopen_s(&fp, argv[i+1], "rb");
		if (fp == NULL) {
			printf("Fail to read the file - %s\n", argv[i + 1]);
			exit(2);
		}
		fread_s(&numArmies, sizeof(UINT), sizeof(UINT), 1, fp);
		printf("%s: %d armys\n", argv[i + 1], numArmies);
		teams[i] = new Team(numArmies, fp);
		fclose(fp);
	}
}