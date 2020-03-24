#include <ctime>


float plus(float a,float b) {
	return a + b;
}
float minus(float a,float b) {
	return a - b;
}
float times(float a,float b) {
	return a * b;
}
float dividedby(float a, float b) {
	return a / b;
}

float gen_random_float(float LO, float HI)
{
    return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
}
