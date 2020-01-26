#include <iostream>
#include <vector>
#include "utils.h"


class Matrix {
public:
    int dims;
    std::vector<int>& shape;
    int *memPtr;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { 
        memPtr = CreateArray(dims, shape);
        Zero();
    }    

    int *CreateArray(int N, std::vector<int>& D) {
        //  Calculate size needed.
        int s = sizeof(int);
        for (int n = 0; n < N; ++n)
            s *= D[n];

        //  Allocate space.
        return (int*) malloc(s);
    }

    void Zero(){
        std::vector<int> input_shape;

        // for(int d = 0; d < dims; ++d){
        for(int v = 0; v < shape[0]; ++v){

            for(int od = 0; od < dims; ++od){
                if(od != 0)
                    for(int ov = 0; ov < shape[od]; ++ov){
                        std::cout << v << ' ';
                        std::cout << ov << std::endl; 
                    }
            }

        }
        std::cout << std::endl;
        // }

        std::cout << "done" << std::endl;
    }

    int GetElement(int I[]){
        if(dims == 0)
            return *memPtr;
        
        int idx = I[0];
        for(size_t d = 1; d < dims; ++d)
            idx = idx * shape[d] + I[d];

        return *(&memPtr[idx]);
    }

    void SetElement(int I[], int val){
        int idx = I[0];
        for(size_t d = 1; d < dims; ++d)
            idx = idx * shape[d] + I[d];

        *(&memPtr[idx]) = val;
    }
};

int main() {

    std::vector<int> params = {2, 2, 2};

    Matrix mat(params);

    int idx_list[2] = {0, 0};
    mat.SetElement(idx_list, 1);
    // std::cout << mat.GetElement(idx_list);


    return 0;
}