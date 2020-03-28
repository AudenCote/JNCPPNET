#include <iostream>
#include <vector>
#include "sum_my_vector.h"

using namespace::std;

int sum_my_vector(vector<int> my_vector)
{
  int my_sum = 0;
  for (auto iv = my_vector.begin(); iv != my_vector.end(); iv++)       
      my_sum += *iv;

  return my_sum;
}