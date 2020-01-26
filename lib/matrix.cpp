#include <iostream>
#include <vector>
#include "utils.h"


class Matrix {
public:
    int dims;
    std::vector<int>& shape;
    int *memPtr;
    int num_vals = 1;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { 
        memPtr = CreateArray(dims, shape);
        for(int val : params) { num_vals *= val; }
        Zero();
    }    

    int *CreateArray(int N, std::vector<int>& D) {

        int s = sizeof(int);

        for (int n = 0; n < N; ++n)
            s *= D[n];

        return (int*) malloc(s);
    }

    void Zero(){

        std::cout << num_vals << std::endl;

        for (int i = 0; i < num_vals; i++)
            memPtr[i] = i;

    }

    int GetElement(int I[]) throw(std::string) {

        if(dims == 0)
            return *memPtr;
        
        int idx = I[0];
        for(int d = 1; d < dims; ++d)
            idx = idx * shape[d] + (I[d]);

        if(idx > num_vals)
            throw "That index is not valid";


        return *(&memPtr[idx]);
    }

    void SetElement(int I[], int val){

        int idx = I[0];
        for(int d = 1; d < dims; ++d)
            idx = idx * shape[d] + (I[d]);

        *(&memPtr[idx]) = val;
    }
};

int main() {

    std::vector<int> params = {2, 2, 2};

    Matrix mat(params);

    int get_idx[3] = {1, 1, 2};
    std::cout << mat.GetElement(get_idx) << std::endl;

    return 0;
}