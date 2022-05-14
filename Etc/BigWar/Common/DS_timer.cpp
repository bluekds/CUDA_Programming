#include "DS_timer.h"
#include "DS_definitions.h"
#include <iostream>

//#ifndef _WIN32
//#define fprintf_s fprintf
//void fopen_s(FILE** fp, char* name, char* mode)
//{
//	*fp = fopen(name, mode);
//}
//#endif

/************************************************************************/
/* Constructor & Destructor                                             */
/************************************************************************/
DS_timer::DS_timer(int _numTimer /* =0 */, int _numCount /* =0 */, bool _trunOn /* =true */){

	turnOn = _trunOn ;
	start_ticks = NULL ; end_ticks = NULL ; totalTicks = NULL ;
	counters = NULL ;
	timerStates = NULL ;
	timerName = NULL ;

	numTimer = numCounter = 0 ;

	setTimerTitle((char*)"DS_timer Report") ;

	setTimer(_numTimer ) ;
	setCounter( _numTimer ) ;

#ifdef _WIN32
	// For windows
	QueryPerformanceFrequency(&ticksPerSecond) ;
#endif
}

DS_timer::~DS_timer(void){
	releaseTimers() ;
	releaseCounters() ;
}

/************************************************************************/
/* Set & Get configurations                                             */
/************************************************************************/

// Memory allocation
void DS_timer::memAllocCounters( void ) {
	releaseCounters() ;
	counters = new UINT[numCounter] ;
	initCounters() ;
}

void DS_timer::memAllocTimers( void ) {
	releaseTimers() ;

	timerStates = new bool[numTimer] ;
	start_ticks = new TIME_VAL[numTimer];
	end_ticks = new TIME_VAL[numTimer];
	totalTicks = new TIME_VAL[numTimer];

	// Initialize
	memset(timerStates, 0, sizeof(bool)*numTimer);
	memset(start_ticks, 0, sizeof(TIME_VAL)*numTimer);
	memset(end_ticks, 0, sizeof(TIME_VAL)*numTimer);
	memset(totalTicks, 0, sizeof(TIME_VAL)*numTimer);

	timerName = new std::string[numTimer] ;
	for ( UINT i = 0 ; i < numTimer ; i++ ) {
		timerName[i].clear() ;
		timerName[i].resize(255) ;
	}
}

// Memory release
void DS_timer::releaseCounters( void ) { DS_MEM_DELETE_ARRAY(counters) ; }
void DS_timer::releaseTimers( void ) {
	//DS_MEM_DELETE(start_ticks) ; DS_MEM_DELETE(end_ticks) ; DS_MEM_DELETE(totalTicks) ;
	DS_MEM_DELETE_ARRAY(timerStates) ;
	DS_MEM_DELETE_ARRAY(start_ticks) ;
	DS_MEM_DELETE_ARRAY(end_ticks) ;
	DS_MEM_DELETE_ARRAY(totalTicks) ;
	DS_MEM_DELETE_ARRAY(timerName) ;
}

// Getters
UINT DS_timer::getNumTimer( void ) { return numTimer ; }
UINT DS_timer::getNumCounter( void ) { return numCounter ; }

// Setters
UINT DS_timer::setTimer( UINT _numTimer ) {
	if ( _numTimer == 0 )
		return 0 ;

	if (_numTimer <= numTimer)
		return numTimer;

	if ( numTimer != 0 ) {

		// Backup
		UINT oldNumTimer = numTimer ;
		TIME_VAL *oldTotalTicks = new TIME_VAL[oldNumTimer];
		memcpy(oldTotalTicks, totalTicks, sizeof(TIME_VAL)*oldNumTimer);

		numTimer = _numTimer ;
		memAllocTimers() ;

		memcpy(totalTicks, oldTotalTicks, sizeof(TIME_VAL)* oldNumTimer);
		delete oldTotalTicks ;
	} else {
		numTimer = _numTimer ;
		memAllocTimers() ;
	}

	return _numTimer ;
}

UINT DS_timer::setCounter( UINT _numCounter ) {

	if (_numCounter == 0 )
		return 0 ;

	if (_numCounter <= numCounter)
		return numCounter;

	if ( numCounter != 0 ) {

		// Backup
		int numOldCounter = numCounter ;
		UINT *oldCounters = new UINT[numOldCounter] ;
		memcpy(oldCounters, counters, sizeof(UINT)*numOldCounter) ;

		numCounter = _numCounter ;
		memAllocCounters() ;

		// Restore
		memcpy(counters, oldCounters, sizeof(UINT)*numOldCounter) ;
		delete oldCounters ;

	} else {
		numCounter = _numCounter ;
		memAllocCounters() ;
	}

	return numCounter ;

}

/************************************************************************/
/* Timer                                                                */
/************************************************************************/
void DS_timer::initTimer( UINT id ) {
	timerStates[id] = TIMER_OFF ;
#ifdef _WIN32
	totalTicks[id].QuadPart = 0 ;
#else
	totalTicks[id].tv_sec = 0;
	totalTicks[id].tv_usec = 0;
#endif
}

void DS_timer::initTimers( void ) {
	for ( UINT i = 0 ; i < numTimer ; i++ ) {
		initTimer(i);
	}
}

void DS_timer::onTimer( UINT id ) {
	if ( turnOn == false )
		return ;

	if ( timerStates[id] == TIMER_ON )
		return ;
#ifdef _WIN32
	QueryPerformanceCounter(&start_ticks[id]) ;
#else
	gettimeofday(&start_ticks[id], NULL);
#endif

	timerStates[id] = TIMER_ON ;
}

void DS_timer::offTimer( UINT id ) {
	if ( turnOn == false )
		return ;

	if ( timerStates[id] == TIMER_OFF )
		return ;

#ifdef _WIN32
	QueryPerformanceCounter(&end_ticks[id]) ;
	totalTicks[id].QuadPart = totalTicks[id].QuadPart + (end_ticks[id].QuadPart - start_ticks[id].QuadPart) ;
#else
	gettimeofday(&end_ticks[id], NULL);
	TIME_VAL period, previous;
	timersub(&end_ticks[id], &start_ticks[id], &period);
	previous = totalTicks[id];
	timeradd(&previous, &period, &totalTicks[id]);
#endif

	timerStates[id] = TIMER_OFF ;
}

double DS_timer::getTimer_ms( UINT id ) {
#ifdef _WIN32
	return ((double)totalTicks[id].QuadPart/(double)ticksPerSecond.QuadPart * 1000) ;
#else
	return (double)(totalTicks[id].tv_sec * 1000 + totalTicks[id].tv_usec / 1000.0);
#endif
}

/************************************************************************/
/* Counter                                                              */
/************************************************************************/
void DS_timer::incCounter( UINT id ) {
	if ( turnOn == false )
		return ;

	counters[id]++ ;
}

void DS_timer::initCounters( void ) {
	if ( turnOn == false )
		return ;

	for ( UINT i = 0 ; i < numCounter ; i++ )
		counters[i] = 0 ;
}

void DS_timer::initCounter( UINT id ) {
	if ( turnOn == false )
		return ;

	counters[id] = 0 ;
}

void DS_timer::add2Counter( UINT id, UINT num ) {
	if ( turnOn == false )
		return ;

	counters[id] += num ;
}

UINT DS_timer::getCounter( UINT id ) {
	if ( turnOn == false )
		return 0;

	return counters[id] ;
}

/************************************************************************/
/*                                                                      */
/************************************************************************/
void DS_timer::printTimer( float _denominator){

	if ( turnOn == false )
		return ;

	//printf("\n*\t DS_timer Report \t*\n") ;
	printf("\n*\t %s \t*\n", timerTitle) ;
	printf("* The number of timer = %d, counter = %d\n", numTimer, numCounter ) ;
	printf("**** Timer report ****\n") ;

	for ( UINT i = 0 ; i < numTimer ; i++ ) {
		if ( getTimer_ms(i) == 0 )
			continue ;
		if ( timerName[i].c_str()[0] == 0 )
			printf("Timer %d : %.5f ms\n", i, getTimer_ms(i) ) ;
		else
			printf("%s : %.5f ms\n", timerName[i].c_str(), getTimer_ms(i) ) ;
	}

	printf("**** Counter report ****\n") ;
	for ( UINT i = 0 ; i < numCounter ;i++ ) {
		if ( counters[i] == 0 )
			continue ;
		printf("Counter %d : %.3f (%d) \n",i, counters[i]/_denominator, counters[i] ) ;
	}

	printf("*\t End of the report \t*\n") ;
}

void DS_timer::printToFile( char* fileName, int _id )
{
	if ( turnOn == false )
		return ;

	FILE *fp ;

	if ( fileName == NULL)
		fopen_s(&fp, "DS_timer_report.txt", "a") ;
	else {
		fopen_s(&fp, fileName, "a") ;
	}

	if ( _id >= 0 )
		fprintf_s(fp,"%d\t", _id) ;

	for ( UINT i = 0 ; i < numTimer ; i++ ) {
		if ( getTimer_ms(i) == 0 )
			continue ;
		fprintf_s(fp, "%.9f", getTimer_ms(i) ) ;
	}

	for ( UINT i = 0 ; i < numCounter ;i++ ) {
		if ( counters[i] == 0 )
			continue ;
		fprintf_s(fp, "%d", counters[i] ) ;
	}

	fprintf_s(fp, "\n") ;

	fclose(fp) ;
}

void DS_timer::printTimerNameToFile( char* fileName )
{
	if ( turnOn == false )
		return ;

	FILE *fp ;

	if ( fileName == NULL)
		fopen_s(&fp, "DS_timer_name.txt", "a") ;
	else {
		fopen_s(&fp, fileName, "a") ;
	}


	for ( UINT i = 0 ; i < numTimer ; i++ ) {
		if ( timerName[i].empty() )
			continue ;
		fprintf_s(fp, "%s\t", timerName[i].c_str() ) ;
	}
	fprintf_s(fp, "\n") ;
	fclose(fp) ;
}
