#include "../Common/BigWar.h"
#include "../Common/DS_timer.h"
#include "Grader.h"

int main(int argc, char** argv) {
	if (argc < 4) {
		printf("Usage: exe inputA inputB outputFile GT_file(optional for grading) TeamID(optional for grading)\n");
		exit(1);
	}

	DS_timer timer(1,1);
	timer.initTimers();
	Pair result[100];

	timer.onTimer(0);
 	// **************************************//
	// Write your code here
	// CAUTION: DO NOT MODITY OTHER PART OF THE main() FUNCTION



	//***************************************//
	timer.offTimer(0);
	timer.printTimer(0);

	// Result validation
	if (argc < 5)
		return 0;

	// Grading mode
	if (argc < 6) {
		printf("Not enough argument for grading\n");
		exit(2);
	}

	Grader grader(argv[4]);
	grader.grading(result);

	FILE* fp = NULL;
	fopen_s(&fp, argv[5], "a");
	if (fp == NULL) {
		printf("Fail to open %s\n", argv[5]);
		exit(3);
	}
	fprintf(fp, "%f\t%d\n", timer.getTimer_ms(0), grader.getNumCorrect());
	fclose(fp);

	return 0;
}