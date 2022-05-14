/** CAUTION: DO NOT TOUCH THIS FILE **/

#include "Grader.h"
#include <string>

Grader::Grader(char* _fileName) {
	strcpy_s(fileName, _fileName);
	fp = NULL;
	corret = 0;
	loadGT();
}

Grader::~Grader()
{
	// no work
}

void Grader::loadGT()
{
	fopen_s(&fp, fileName, "r");
	if (fp == NULL) {
		printf("Fail to open '%s'\n", fileName);
		exit(1);
	}

	for (int i = 0; i < NUM_ANSWER; i++) {
		fscanf_s(fp, "%d %d %f\n", &GT[i].A, &GT[i].B, &GT[i].dist);
		//GT[i].print();
	}

	fclose(fp);
}


void Grader::grading(Pair* _answer) {
	
	for (int i = 0; i < NUM_ANSWER; i++) {
		if (_answer[i].A == GT[i].A && _answer[i].B == GT[i].B)
			corret++;
	}

	printf("Corret anwers: %d/%d (%.2f%%)\n", corret, NUM_ANSWER, corret*100.f / (float)NUM_ANSWER);
}

int Grader::getNumCorrect()
{
	return corret;
}
