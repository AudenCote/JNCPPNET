#include <iostream>
#include <vector>
#include "utils.h"
#include <random>
#include <math.h>


class Matrix {
private:

    int *create_array(int N, std::vector<int>& D) {
        int s = sizeof(float);

        for (int n = 0; n < N; ++n)
            s *= D[n];

        return (int*) malloc(s);
    }

    static Matrix *elementwise_operation(Matrix& mat1, Matrix& mat2, float (*func)(float, float)) {

        try{
            if(mat1.num_vals != mat2.num_vals)
                throw std::invalid_argument("Matrix dimensions do not match");
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function ElementwiseAddition" << std::endl << std::endl;
        }

        Matrix *outmat = new Matrix(mat1.shape);

        for (int i = 0; i < outmat->num_vals; ++i)
            outmat->memPtr[i] = func(mat1.memPtr[i], mat2.memPtr[i]);

        return outmat;
    }

    float sigmoid_operator(float x) {
        return 1/(1+exp(-(x)));
    }

    float sigmoid_prime_operator(float x) {
         return 1/(1+exp(-(x)))*(1-(1/(1+exp(-(x)))));
    }
    
public:
    int dims;
    std::vector<int>& shape;
    int num_vals = 1;
    int *memPtr;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { 
        memPtr = create_array(dims, shape);
        for(int val : params) { num_vals *= val; }
        Zero();
    }    

    void Zero() {
        for (int i = 0; i < num_vals; ++i)
            memPtr[i] = 0.0f;
    }

    void Randomize() {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1, 1);

        for (int i = 0; i < num_vals; ++i)
            memPtr[i] = distribution(generator);
    }

    float GetVal(std::initializer_list<int> init_list) {

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

        try{
            if(dims == 0)
                return *memPtr;

            for(int d = 0; d < dims; ++d)
                if(I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing");
            if(std::sizeof(I) != dims)
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetChunk function instead of GetVal --");

            int idx = I[0];
            for(int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);
            if(idx > num_vals)
                throw std::invalid_argument("Invalid Indexing");
            return memPtr[idx];

        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function GetElement" << std::endl << std::endl;
            return 0;
        }
    }

    Matrix *GetChunk(std::initializer_list<int> init_list) {

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

        try{
            if(dims == 0)
                return nullptr;

            for(int d = 0; d < dims; ++d)
                if(I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing");

            if(sizeof(I) >= dims)
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetVal function instead of GetChunk --");


            int start_idx = 0;
            int denomenator = 1;
            for(int i = 0; i < std::sizeof(I); ++i){
                denomenator *= shape[i];
                start_idx += I[i]*shape[i];
            }

            //ADD PROTECTION FOR SEGMENTATION FAULTS

            //make sure start_idx is the right start point - plus or minus one?
            std::vector<float> out_mat_vals;
            for(int i = start_idx; i < num_vals/denomenator; ++i)
                out_mat_vals.push_back(memPtr[i]);

            std::vector<int> out_mat_shape;
            Matrix out_mat = new Matrix(out_mat_shape);
            for(int v = 0; v < out_mat.num_vals; ++v)
                out_mat.memPtr[v] = out_mat_vals[v];

            return &out_mat;

        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function GetElement" << std::endl << std::endl;
            return NULL;
        }

    }

    void Set(std::initializer_list<int> init_list, float val) {

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

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
            std::cout << std::endl << e.what() << " in function SetElement" << std::endl << std::endl;
        }
    }

    void Add(float val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] += val;
    }

    void Subtract(float val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] -= val;
    }

    void Multiply(float val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] *= val;
    }

    void Divide(float val) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] /= val;
    }

    void Square() {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] = memPtr[i]*memPtr[i];
    }

    static Matrix *DotProduct(Matrix& mat1, Matrix& mat2) {

        try{
            if(mat1.dims != 2 || mat2.dims != 2)
                throw std::invalid_argument("Invalid matrix dimensions");
            if(mat1.shape[1] != mat2.shape[0])
                throw std::invalid_argument("Matrix dimensions don't match: Invalid matrix dimensions");
        }
        catch(const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function DotProduct" << std::endl << std::endl;
        }

        int row1 = mat1.shape[0],  col1 = mat1.shape[1], row2 = mat2.shape[0], col2 = mat2.shape[1];
        int size = row1*col2;

        std::vector<int> out_shape = {row1, col2};
        Matrix *out = new Matrix(out_shape);

        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col2; j++) {
                int total = 0;
                for (int k = 0; k < col1; k++)
                    total = total + mat1.memPtr[i * col1 + k] * mat2.memPtr[k * col2 + j];
                out->memPtr[i*col2+j] = total;
            }
        }

        return out;

    }

    static Matrix *ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, plus);
    }
    static Matrix *ElementwiseSubtraction(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, minus);
    }
    static Matrix *ElementwiseMultiplication(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, times);
    }
    static Matrix *ElementwiseDivision(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, dividedby); 
    }

    float Sum() {
        float sum = 0;
        for (int i = 0; i < num_vals; i++)
            sum += memPtr[i];
        return sum;
    }

    void Map(float (*func)(float)) {
        for (int i = 0; i < num_vals; i++)
            memPtr[i] = func(memPtr[i]);
    }

    static void Sigmoid(Matrix* mat) {
        mat->Map(sigmoid_operator);
    }

    static void SigmoidPrime(Matrix* mat) {
        mat->Map(sigmoid_prime_operator);
    }

    static Matrix *Transpose(Matrix* mat) {
        if(mat->dims == 2){
            Matrix* out_mat = new Matrix(mat->shape[1], mat->shape[0])
            for(int i = 0; i < mat->num_vals; ++i){
                if(i % mat->shape[0] == 0){
                    for(int j = 0; j < mat->shape[0]; ++j){
                        out_mat->Set({j, i/mat->shape[0]}, mat->GetVal({i/mat->shape[0], j}));
                    }
                }
                
            }
            return out_mat;
        }else {
            std::cout << "Cannot transpose matrix of invalid dimensions" << std::endl;
            return mat;
        }
    }
};