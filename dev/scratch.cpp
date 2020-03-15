#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

float gen_random_float(float LO, float HI)
{
    return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
}

main() {
    for(int i = 0; i < 100; ++i) cout << gen_random_float(-1, 1) << endl;
}