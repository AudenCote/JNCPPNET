#include <iostream>
#include <vector>
#include "utils.h"


class Matrix {
private:
    int *memPtr;

    int *CreateArray(int N, std::vector<int>& D) {

        int s = sizeof(int);

        for (int n = 0; n < N; ++n)
            s *= D[n];

        return (int*) malloc(s);
    }

    static Matrix *ElementwiseOperation(Matrix& mat1, Matrix& mat2, int (*func)(int, int)) {

        try{
            if(mat1.num_vals != mat2.num_vals)
                throw std::invalid_argument("Matrix dimensions do not match");
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e << " in function ElementwiseAddition" << std::endl;
        }

        Matrix outmat(mat1.shape);

        for (int i = 0; i < outmat.num_vals; ++i)
            outmat.memPtr[i] = func(mat1.memPtr[i], mat2.memPtr[i]);

        return &outmat;
    }
    
public:
    int dims;
    std::vector<int>& shape;
    int num_vals = 1;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { 
        memPtr = CreateArray(dims, shape);
        for(int val : params) { num_vals *= val; }
        Zero();
    }    

    void Zero() {
        for (int i = 0; i < num_vals; ++i)
            memPtr[i] = 0;
    }

    int GetElement(int I[]) {
        try{
            if(dims == 0)
                return *memPtr;

            for(int d = 0; d < dims; ++d)
                if(I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing");
            
            int idx = I[0];
            for(int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);

            if(idx > num_vals)
                throw std::invalid_argument("Invalid Indexing");


            return memPtr[idx];
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e << " in function GetElement" << std::endl;
        }
    }

    void SetElement(int I[], int val) {
        try{
            for(int d = 0; d < dims; ++d)
                if(I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing");

            int idx = I[0];
            for(int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);

            if(idx > num_vals)
                throw std::invalid_argument("Invalid Indexing");

            memPtr[idx] = val;
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e << " in function SetElement" << std::endl;
        }
    }

    void Add(int val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] += val;
    }

    void Subtract(int val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] -= val;
    }

    void Multiply(int val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] *= val;
    }

    void Divide(int val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] /= val;
    }

    static Matrix *DotProduct(Matrix& mat1, Matrix& mat2) {

        try{
            if(mat1.dims != 2 || mat2.dims != 2)
                throw std::invalid_argument("Invalid matrix dimensions");
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e << " in function DotProduct" << std::endl;
        }

        std::vector<std::vector<int>> m1vec, m2vec, outvec;
        std::vector<int> outshape, lstemp1, lstemp2;

        for (int i = 0; i < mat1.num_vals; i++)
            lstemp1.push_back(mat1.memPtr[i]);
        for (int i = 0; i < mat2.num_vals; i++)
            lstemp2.push_back(mat2.memPtr[i]);

        for(int d = 0; d < 2; ++d) {
            m1vec = {};
            for(int j = 0; j < mat1.num_vals; ++j) {
                if(j % mat1.shape[d] == 0) {
                    for(int i = j; i < mat1.shape[d]; ++i) {
                        m1vec.push_back(lstemp1[j]);
                    }
                }
            }
            lstemp1 = m1vec;
            

            m2vec = {};
            for(int j = 0; j < mat2.num_vals; ++j) {
                if(j % mat2.shape[d] == 0) {
                    for(int i = j; i < mat2.shape[d]; ++i) {
                        m2vec.push_back(lstemp1);
                    }
                }
            }
            lstemp2 = m2vec;
        }

        std::vector<int> temprow;
        int total = 0;

        for(int i = 0; i < m1vec.size(); ++i) {
            temprow = {};
            for(int j = 0; j < m2vec[0].size(); ++j) {
                total = 0;
                for(int k = 0; k < m2vec.size(); ++k) {
                    total = total + m1vec[i][k] + m2vec[k][j];
                }
                temprow.push_back(total);
            }
            outvec.push_back(temprow);
        }

        outshape = {outvec.size(), outvec[0].size()};
        Matrix outmat(outshape);
        return &outmat;
    }

    Matrix *ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        return ElementwiseOperation(mat1, mat2, plus);
    }
    Matrix *ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        return ElementwiseOperation(mat1, mat2, minus);
    }
    Matrix *ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        return ElementwiseOperation(mat1, mat2, times);
    }
    Matrix *ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        //return ElementwiseOperation(mat1, mat2, dividedby); //sort out float/int issue
    }

    float Sum() {
        float sum = 0;
        for (int i = 0; i < num_vals; i++)
            sum += memPtr[i];
        return sum;
    }

    void Map(int (*func)(int)) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] = func(memPtr[i]);
    }



};




int main() {

    std::vector<int> params = {2, 2, 2};

    Matrix mat1(params);

    int get_idx[3] = {1, 1, 2};
    std::cout << mat1.GetElement(get_idx) << std::endl;

    return 0;
}