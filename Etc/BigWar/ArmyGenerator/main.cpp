#include "../Common/BigWar.h"
#include "ArmyGenerator.h"
#include <iostream>
#include <fstream>

//#define _DEBUG
using namespace std;

Army genArmy(UINT _min = RANGE_MIN, UINT _max = RANGE_MAX);

int main(int argc, char** argv)
{
	if (argc < 3) {
		printf("Usage: exe (# of armys) (output file name)");
		exit(1);
	}

	// Check request
	unsigned int numArmy = atoi(argv[1]);
	if (numArmy > INT_MAX) {
		printf("The maximum number of armys is %d\n", INT_MAX);
		exit(2);
	}
	char* outputFile = argv[2];
	printf("# of armys = %d, outputFile = %s\n", numArmy, outputFile);

	// Generation
	ofstream writeFile;
	writeFile.open(outputFile, std::ios::binary);
	writeFile.write((char*)&numArmy, sizeof(UINT));

	Army army;
	ArmyGenerator gen(RANGE_MIN, RANGE_MAX);
	for (unsigned int i = 0; i < numArmy; i++) {
		army = gen.genArmy();
		#ifdef _DEBUG
		army.print();
		#endif
		writeFile.write((char*)army.pos, sizeof(POS_TYPE) * ARMY_DIMENSION);
		if (numArmy > 10 && (i % (numArmy / 10)) == 0)
			printf("%d%% (%d/%d)\n", (i * 100 /numArmy), i, numArmy);
	}
	writeFile.close();
	printf("Generataion was complemtead!\n");
	
	return 0;
}