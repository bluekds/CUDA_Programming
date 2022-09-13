#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#define	OS_WINDOWS	0
#define OS_LINUX	1

#ifdef _WIN32
#define _TARGET_OS	OS_WINDOWS
#else
	#ifndef nullptr
	#define nullptr NULL
	#endif
	#define _TARGET_OS	OS_LINUX
#endif

/************************************************************************/
/* OS dependet function                                                 */
/************************************************************************/
#if _TARGET_OS == OS_WINDOWS
//	#define	_SPRINT	sprintf_s
#define	_STRTOK strtok_s

#define EXIT_WIHT_KEYPRESS {std::cout << "Press any key to exit..."; getchar(); exit(0);}

#define SPLIT_PATH(_path,_result)	\
	_splitpath_s(_path, _result.drive, 255, _result.dir, 255, _result.filename, 255, _result.ext, 255)


#elif _TARGET_OS == OS_LINUX
#include <libgen.h>
#include <inttypes.h>

#define	_STRTOK strtok_r

#define EXIT_WIHT_KEYPRESS {std::cout << "Program was terminated!"; exit(0);}

#define sprintf_s	sprintf
#define scanf_s		scanf
#define fprintf_s	fprintf

#define __int64		int64_t

#define fopen_s(fp, name, mode) (*fp = fopen(name, mode))

#endif

/************************************************************************/
/* Defines                                                              */
/************************************************************************/

// *********** data size
#define	_1K_	1024
#define _1M_	(_1K_*_1K_)
#define _1G_	(_1M_*_1K_)

#define CHAR_STRING_SIZE	255

/************************************************************************/
/* Type definitions                                                     */
/************************************************************************/
typedef unsigned int UINT ;

/************************************************************************/
/* Macro functions                                                      */
/************************************************************************/
#define DS_MEM_DELETE(a)	\
	if (a != NULL) {		\
		delete a ;			\
		a = NULL ;			\
	}

#define DS_MEM_DELETE_ARRAY(a)	\
	if (a != NULL) {		\
	delete [] a ;			\
	a = NULL ;			\
	}

#define RANGE_MIN 0
#define RANGE_MAX 1

#define MATCHED_STRING 0

#ifndef VTK_RANGE_MIN
#define VTK_RANGE_MIN	0
#define VTK_RANGE_MAX	1
#endif

// Print
#define PRINT_LINE_INFO printf("%s, line %d", __FILE__, __LINE__)
#define PRINT_ERROR_MSG(_msg) {PRINT_LINE_INFO; printf(" at "); printf(_msg);}

// Single loops
#define LOOP_I(a) for(int i=0; i<a; i++)
#define LOOP_J(a) for(int j=0; j<a; j++)
#define LOOP_K(a) for(int k=0; k<a; k++)
#define LOOP_INDEX(index, end) for (int index = 0 ; index < end ; index++)
#define LOOP_INDEX_START_END(index, start, end) for (int index = start ; index < end ; index++)

// Multiple loops
#define LOOP_J_I(b, a) LOOP_J(b) LOOP_I(a)
#define LOOP_K_J_I(c,b,a) for(int k=0; k<c; k++) LOOP_J_I(b,a)

//
#ifndef SWAP
template<class T>
void SWAP(T &a, T &b){
	T tmp = a;
	a = b;
	b = tmp;
}
#endif

// 
#ifndef MIN
#define MIN(a,b) (a > b ? b : a)
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Index converter

#define INDEX2X(_ID,_W) (_ID%_W)
#define INDEX2Y(_ID,_W) (_ID/_W)
#define INDEX2ID(_ID,_X,_Y,_W) {_X=INDEX2X(_ID,_W);_Y=INDEX2Y(_ID_,_W);}
#define ID2INDEX(_W,_X,_Y) (_Y*_W+_X)
#define PTR2ID(_type, _target, _base) ((_type*)_target - (_type*)_base)

// Memory allocation and release
#ifndef SAFE_DELETE
#define	SAFE_DELETE(p) {if(p!=NULL) delete p; p=NULL;}
#endif

#ifndef SAFE_DELETE_ARR
#define	SAFE_DELETE_ARR(p) {if(p!=NULL) delete [] p; p=NULL;}
#endif

#define SAFE_NEW(p, type, size) {\
	try {p = new type[size];}	\
	catch(std::bad_alloc& exc) \
	{ printf("[%s, line %d] fail to memory allocation - %.2f MB requested\n", __FILE__, __LINE__, (float)(sizeof(type)*size)/_1M_);	\
	EXIT_WIHT_KEYPRESS }\
	}

template<class T>
void memsetZero(T** p, long long size = 0) {
	if (*p != NULL)
		memset(*p, 0, sizeof(T)*size);
}

template<class T>
void allocNinitMem(T** p, long long size, double *memUsage = NULL) {
	*p = new T[size];
	//SAFE_NEW(*p, T, size);
	memset(*p, 0, sizeof(T)*size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T)*size;
	}
}

#define SAFE_MEMCPY(_dst, _src, _type, _size){ \
	if(_dst == nullptr || _src == nullptr ) \
		printf("[%s, line %d] fail to memcpy (dst = %x, src = %x)\n", __FILE__, __LINE__, _dst, _src);	\
		exit(-1);	\
	memcpy(_dst, _src, sizeof(_type)*_size);\
}

// VTK related
#ifndef SAFE_DELETE_VTK
#define SAFE_DELETE_VTK(p) {if(p!=NULL) p->Delete(); p=NULL;}
#endif

#ifndef VTK_IS_NOERROR
//#include "DS_common_def.h"
#define VTK_IS_NOERROR(p) (p->GetErrorCode()==vtkErrorCode::NoError ? true : false)
#endif

/************************************************************************/
/* Data structures	                                                    */
/************************************************************************/
typedef struct {
	std::string input;
	std::string output;
} nameMatch;

typedef struct {
	char drive[255];
	char dir[255];
	char filename[255];
	char ext[255];
} filePathSplit;
