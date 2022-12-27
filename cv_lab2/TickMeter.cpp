#include "TickMeter.h"

TickMeter::TickMeter() { reset(); }
int64_t TickMeter::getTimeTicks() const { return sumTime; }
double TickMeter::getTimeMicro() const { return  getTimeMilli()*1e3; }
double TickMeter::getTimeMilli() const { return getTimeSec()*1e3; }
double TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64_t TickMeter::getCounter() const { return counter; }
void TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }

void TickMeter::start(){ startTime = cv::getTickCount(); }
void TickMeter::stop()
{
	int64 time = cv::getTickCount();
	if (startTime == 0)
		return;
	++counter;
	sumTime += (time - startTime);
	startTime = 0;
}

std::ostream& operator<< (std::ostream& out, const TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }
