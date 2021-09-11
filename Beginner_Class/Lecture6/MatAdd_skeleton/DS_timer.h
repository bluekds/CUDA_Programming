//#pragma once
#ifndef _DS_TIMER_H
#define _DS_TIMER_H

#include <string> // std string

#ifndef UINT
typedef unsigned int UINT;
#endif

#ifdef _WIN32
	// For windows
	#include <Windows.h>
	typedef LARGE_INTEGER	TIME_VAL;
#else
	// For Unix/Linux
	#include <stdio.h>
	#include <stdlib.h>
	#include <sys/time.h>
	#include <string.h>	// c string
	typedef struct timeval	TIME_VAL;
#endif

#define TIMER_ON	true
#define TIMER_OFF	false

class DS_timer
{
private :

	bool turnOn	;

	UINT numTimer	;
	UINT numCounter ;

	// For timers
	bool*			timerStates ;
	TIME_VAL	ticksPerSecond;
	TIME_VAL	*start_ticks;
	TIME_VAL	*end_ticks;
	TIME_VAL	*totalTicks;

	char		timerTitle[255] ;
	std::string *timerName ;

	// For counters
	UINT *counters ;

	void memAllocCounters ( void ) ;
	void memAllocTimers ( void ) ;
	void releaseCounters ( void ) ;
	void releaseTimers ( void ) ;

public:
	DS_timer(int _numTimer = 1, int _numCount = 1, bool _trunOn = true );
	~DS_timer(void);

	// For configurations
	inline void timerOn ( void ) { turnOn = TIMER_ON ; }
	inline void timerOff ( void ) { turnOn = TIMER_OFF ; }

	UINT getNumTimer( void ) ;
	UINT getNumCounter ( void ) ;
	UINT setTimer ( UINT _numTimer ) ;
	UINT setCounter ( UINT _numCounter ) ;

	// For timers

	void initTimer(UINT id) ;
	void initTimers ( void );
	void onTimer(UINT id) ;
	void offTimer(UINT id ) ;
	double getTimer_ms(UINT id) ;

	void setTimerTitle ( char* _name ) { memset(timerTitle, 0, sizeof(char)*255) ; memcpy(timerTitle, _name, strlen(_name)) ; }

	void setTimerName (UINT id, std::string &_name) { timerName[id] = _name ; }
	void setTimerName (UINT id, char* _name) { timerName[id] = _name ;}

	// For counters

	void incCounter(UINT id) ;
	void initCounters( void ) ;
	void initCounter(UINT id) ;
	void add2Counter( UINT id, UINT num ) ;
	UINT getCounter ( UINT id ) ;

	// For reports

	void printTimer ( float _denominator = 1 ) ;
	void printToFile ( char* fileName, int _id = -1 ) ;
	void printTimerNameToFile ( char* fileName ) ;
} ;

#endif