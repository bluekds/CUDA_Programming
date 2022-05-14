#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>

#define RANGE_MIN (0)
#define RANGE_MAX (1024*1024)
#define	ARMY_DIMENSION 3
#define EPSILON (10e-6)

typedef unsigned int UINT;
typedef float POS_TYPE;

class Army {
public:
	static POS_TYPE dist(Army _A, Army _B) {
		POS_TYPE d = 0;
		for (int i = 0; i < ARMY_DIMENSION; i++) {
			d += (_A.pos[i] - _B.pos[i]) * (_A.pos[i] - _B.pos[i]);
		}
		return sqrt(d);
	}
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

class Pair {
public:
	UINT A, B;
	POS_TYPE dist;
public:
	static bool compare(Pair _pair1, Pair _pair2) {
		if (std::abs((POS_TYPE)(_pair1.dist - _pair2.dist)) < EPSILON) { // equidistant  
			if (_pair1.A == _pair2.A)
				return _pair1.B < _pair2.B;
			return _pair1.A < _pair2.A;
		}
		return _pair1.dist < _pair2.dist;
	}
public:
	Pair() { A = B = 0; dist = 0; }

	bool operator >(Pair& ref) {return !compare(*this, ref);}
	bool operator <(Pair& ref) {return compare(*this, ref);}

	void set(UINT _a, UINT _b, POS_TYPE _dist) { A = _a; B = _b; dist = _dist; }
	void print(void) { printf("(%u,%u): %f\n", A, B, dist); }
};