#include "../Common/BigWar.h"
#include "Team.h"
#include <vector>
#include <algorithm>

using namespace std;
#define NUM_RESULTS 100
#define LAST_PAIR (NUM_RESULTS-1)

int main(int argc, char** argv)
{
	if (argc < 4) {
		printf("Usage: exe inputA inputB outputFile\n");
		exit(1);
	}

	FILE* fp = NULL;
	UINT numArmies = 0;	
	Team *teams[2];

	for (int i = 0; i < 2; i++) { // for each team
		fopen_s(&fp, argv[i+1], "rb");
		if (fp == NULL) {
			printf("Fail to read the file - %s\n", argv[i + 1]);
			exit(2);
		}
		fread_s(&numArmies, sizeof(UINT), sizeof(UINT), 1, fp);
		printf("%s: %d armys\n", argv[i + 1], numArmies);
		teams[i] = new Team(numArmies, fp);
		teams[i]->loadTeam();
		fclose(fp);
	}

	vector<Pair> result(NUM_RESULTS);
	for (int i = 0; i < NUM_RESULTS; i++)
		result[i].dist = RANGE_MAX;

	Pair curPair;
	for (int iA = 0; iA < teams[0]->numArmies; iA++) {
		for (int iB = 0; iB < teams[1]->numArmies; iB++) {
			curPair.set(iA, iB, Army::dist(teams[0]->armies[iA], teams[1]->armies[iB]));
			if (result[LAST_PAIR] > curPair) {
				result[LAST_PAIR] = curPair;
				sort(result.begin(), result.end(), Pair::compare);
			}
		}
	}

	// Write the result
	FILE* wfp = NULL;
	fopen_s(&wfp, argv[3], "w");
	if (wfp == NULL) {
		printf("Fail to open the output file\n");
		exit(2);
	}
	for (int i = 0; i < NUM_RESULTS; i++) {
		result[i].print();
		fprintf(wfp, "%d %d %.2f\n", result[i].A, result[i].B, result[i].dist);
	}
	fclose(wfp);
}