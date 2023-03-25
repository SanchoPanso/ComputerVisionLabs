#include <iostream>
#include <cmath>
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/utility.hpp>


class TickMeter {
public:
	TickMeter();
	void start();
	void stop();

	int64 getTimeTicks() const;
	double getTimeMicro() const;
	double getTimeMilli() const;
	double getTimeSec()   const;
	int64 getCounter() const;

	void reset();
private:
	int64 counter;
	int64 sumTime;
	int64 startTime;
};

std::ostream& operator << (std::ostream& out, const TickMeter& tm);
