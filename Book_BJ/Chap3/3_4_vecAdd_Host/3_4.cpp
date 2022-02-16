#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

int main(void)
{
	int* a, * b, * c;

	int memSize = sizeof(int) * NUM_DATA;
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);

	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	for (int i = 0; i < NUM_DATA; i++)
		c[i] = a[i] + b[i];

	delete[] a; delete[] b; delete[] c;
}
