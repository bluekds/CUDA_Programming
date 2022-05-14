/** CAUTION: DO NOT TOUCH THIS FILE **/
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "../Common/BigWar.h"

#define	NUM_ANSWER 100

class Grader {
private:
	FILE* fp;
	char fileName[255];
	Pair GT[NUM_ANSWER];
	int corret;
private:
	void loadGT();
public:
	Grader(char* _fileName);
	~Grader();

	void grading(Pair* _answer);
	int getNumCorrect();
};