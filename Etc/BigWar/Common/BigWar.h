#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RANGE_MIN (0)
#define RANGE_MAX (1024*1024)
#define	ARMY_DIMENSION 3

typedef unsigned int UINT;
typedef float POS_TYPE;

class Army {
public:
	Army() { memset(this->pos, 0, sizeof(POS_TYPE) * ARMY_DIMENSION); }
	~Army(){}
	void print() {
		for (int i = 0; i < ARMY_DIMENSION; i++)
			printf("%f ", pos[i]);
		printf("\n");
	}
public:
	union {
		POS_TYPE pos[ARMY_DIMENSION];
#if ARMY_DIMENSION == 3
		struct {
			POS_TYPE x;
			POS_TYPE y;
			POS_TYPE z;
		};
#endif
	};
};