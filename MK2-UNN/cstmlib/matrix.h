#ifndef MATRIX_INCLUDE
#define MATRIX_INCLUDE

#include <iostream>
#include <vector>
#include "utils.h"
#include <random>
#include <math.h>
#include <memory>
#include <ctime>
#include "log.h"


class Matrix {
private:
    static std::shared_ptr<Matrix> elementwise_operation(Matrix& mat1, Matrix& mat2, float (*func)(float, float)) {

        try {
            if (mat1.num_vals != mat2.num_vals)
                throw std::invalid_argument("Matrix dimensions do not match in function ElementwiseAddition");
        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
        }

        std::shared_ptr<Matrix> outmat = std::make_shared<Matrix>(mat1.shape);

        for (int i = 0; i < outmat->num_vals; ++i)
            outmat->matrix_values[i] = func(mat1.matrix_values[i], mat2.matrix_values[i]);

        return outmat;
    }

    static float sigmoid_operator(float x) {
        return 1 / (1 + exp(-(x)));
    }

    static float sigmoid_prime_operator(float x) {
        return 1 / (1 + exp(-(x))) * (1 - (1 / (1 + exp(-(x)))));
    }

    static float relu_operator(float x) {
        if (x < 0) {
            x = 0;
        }

        return x;
    }

    static float categorical_cross_entropy_operator(float o, float t) {
        return -o * log(t); //log function is natural, log10() is common
    }

    void init_zeroes() {
        for (int i = 0; i < num_vals; ++i) {
            matrix_values.push_back(0.0f);
        }
    }

    void set_chunk_template(const std::initializer_list<int> init_list, std::vector<float>& vector_to_insert) {
        std::vector<int> I;
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            for (int d = 0; d < dims; ++d)
                if (I[d] >= shape[d] || I[d] < 0)
                    //std::cout << d << " " << I[d] << " " << shape[d] << std::endl;
                    throw std::invalid_argument("Invalid Indexing Code 0 in function SetChunk");

            int start_idx = 0;
            for (int i = 0; i < init_list.size(); ++i) {
                int mult_dims = 1;
                mult_dims *= I[i];
                for (int j = i + 1; j < dims; ++j) {
                    mult_dims *= shape[j];
                }
                start_idx += mult_dims;
            }

            if (start_idx + vector_to_insert.size() > num_vals) {
                throw std::invalid_argument("Invalid Indexing Code 1 in function SetChunk");
            }

            for (int i = start_idx; i < vector_to_insert.size() + start_idx; i++) {
                matrix_values[i] = vector_to_insert[i - start_idx];
            }


        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
        }
        catch (const std::logic_error& e) {
            Logger::Error(e.what());
        }
    }

public:
    int dims;
    std::vector<int> shape;
    int num_vals = 1;
    std::vector<float> matrix_values;

    Matrix(std::vector<int> params) : dims(params.size()), shape(params) {
        for (int val : params) { num_vals *= val; }
        init_zeroes();
    }

    ~Matrix() { }

    void Zero() {
        for (int i = 0; i < num_vals; ++i) {
            matrix_values[i] = 0.0f;
        }
    }

    void Randomize() {
        for (int i = 0; i < num_vals; ++i) {
            matrix_values[i] = gen_random_float(-1, 1);
        }
    }

    float GetVal(std::initializer_list<int> init_list) {

        std::vector<int> I;
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            if (dims == 0)
                return 0;

            for (int d = 0; d < dims; ++d)
                if (I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing in function GetVal");
            if (init_list.size() != dims) {
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetChunk function instead of GetVal -- in function GetVal");
            }

            int idx = I[0];
            for (int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);
            if (idx > num_vals)
                throw std::invalid_argument("Invalid Indexing in function GetVal");
            return matrix_values[idx];

        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
            return 0;
        }
    }

    std::shared_ptr<Matrix> GetChunk(std::initializer_list<int>& init_list) {

        std::vector<int> I;
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            if (dims == 0)
                return nullptr;

            for (int d = 0; d < init_list.size(); ++d) {
                if (I[d] >= shape[d] || I[d] < 0) {
                    throw std::invalid_argument("Invalid Indexing in function GetChunk");
                }
            }

            if (init_list.size() >= dims) {
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetVal function instead of GetChunk -- in function GetChunk");
            }

            int start_idx = 0;
            for (int i = 0; i < init_list.size(); ++i) {
                int mult_dims = 1;
                mult_dims *= I[i];
                for (int j = i + 1; j < dims; ++j) {
                    mult_dims *= shape[j];
                }
                start_idx += mult_dims;
            }

            std::vector<float> out_mat_vals;
            for (int i = start_idx; i < num_vals; ++i) {
                out_mat_vals.push_back(matrix_values[i]);
            }

            std::vector<int> out_mat_shape;
            for (int i = init_list.size(); i < dims; ++i) {
                out_mat_shape.push_back(shape[i]);
            }

            std::shared_ptr<Matrix> out_mat = std::make_shared<Matrix>(out_mat_shape);
            for (int i = 0; i < out_mat->num_vals; ++i) {
                out_mat->matrix_values[i] = out_mat_vals[i];
            }

            return out_mat;
        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
            return nullptr;
        }
    }

    std::shared_ptr<Matrix> GetChunk(my_misc_utils::better_initializer_list<my_misc_utils::better_initializer_list<int>>& init_list) {

        //in each input vector is a series of numbers alternating start and end vales for each chunk in each dim, each dim being an input vector (listed in THE input vector, init_list)


        try {
            //GETTING CHUNK

            for (int i = 0; i < init_list.size(); i++) {
                if (init_list[i].size() > 2 || init_list[i].size() < 1) {
                    throw std::invalid_argument("Invalid input chunk dimensions - make sure the number of values in each dimension is at least one and no more than two - the start and stop values in each dimension, or the single value in each dimension. See docs for more info. Exception thrown in function Matrix::GetChunk()");
                }
            }

            std::vector<int> output_matrix_shape;
            std::vector<float> output_matrix_values;

            for (int i = 0; i < init_list.size(); i++) {
                if (init_list[i].size() == 1) {
                    output_matrix_shape.push_back(1);
                }
                else {
                    output_matrix_shape.push_back(abs(init_list[i][init_list[i].size() - 1] - init_list[i][0]));
                }
            }

            for (int n = 0; n < num_vals; ++n) {
                int num_matching_starting_bounds = 0;
                int num_matching_ending_bounds = 0;

                std::vector<int> unraveled = UnravelIndex(n, shape);

                for (int d = 0; d < dims; ++d) {
                    std::vector<int> starting_values;
                    std::vector<int> ending_values;

                    for (int c = 0; c < init_list[d].size(); c++) {
                        if (c % 2 == 0) {
                            starting_values.push_back(init_list[d][c]);
                        }
                        else {
                            ending_values.push_back(init_list[d][c]);
                        }
                    }

                    for (int v : starting_values) {
                        if (unraveled[d] >= v) {
                            num_matching_starting_bounds += 1;
                            break;
                        }
                    }
                    for (int v : ending_values) {
                        if (unraveled[d] < v) {
                            num_matching_ending_bounds += 1;
                            break;
                        }
                    }
                }

                if (num_matching_starting_bounds + num_matching_ending_bounds == dims*2) {
                    output_matrix_values.push_back(matrix_values[n]);
                }
            }

            std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>(output_matrix_shape);
            output_matrix->matrix_values = output_matrix_values;
            return output_matrix;
        }
        catch(std::invalid_argument e){
            Logger::Error(e.what());
        }
    }

    void SetVal(const std::initializer_list<int> init_list, float val) {

        std::vector<int> I;
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            for (int d = 0; d < dims; ++d)
                if (I[d] >= shape[d] || I[d] < 0)
                    //std::cout << d << " " << I[d] << " " << shape[d] << std::endl;
                    throw std::invalid_argument("Invalid Indexing Code 0 in function Set");

            int idx = I[0];
            for (int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);

            if (idx > num_vals)
                throw std::invalid_argument("Invalid Indexing Code 1 in function Set");

            matrix_values[idx] = val;
        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
        }
    }

    void SetChunk(const std::initializer_list<int> init_list, std::vector<float>& vector_to_insert) {
        set_chunk_template(init_list, vector_to_insert);
    }

    void SetChunk(const std::initializer_list<int> init_list, Matrix& matrix_to_insert) {
        set_chunk_template(init_list, matrix_to_insert.matrix_values);
    }

    static std::shared_ptr<Matrix> Reshape(Matrix& input_matrix, std::initializer_list<int> init_list) {

        int shape_product = 1;
        for (int v : init_list) shape_product *= v;

        if (input_matrix.num_vals != shape_product) {
            Logger::Error("Output and input shapes do not match in function Matrix::Reshape()");
            return nullptr;
        }

        else {
            std::vector<int> output_shape;
            std::copy(init_list.begin(), init_list.end(), output_shape);
            std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>(output_shape);

            output_matrix->matrix_values = input_matrix.matrix_values;

            return output_matrix;
        }
    }

    void Add(float val) {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] += val;
    }

    void Subtract(float val) {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] -= val;
    }

    void Multiply(float val) {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] *= val;
    }

    void Divide(float val) {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] /= val;
    }

    void Square() {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] = matrix_values[i] * matrix_values[i];
    

    static std::shared_ptr<Matrix> DotProduct(Matrix& mat1, Matrix& mat2) {

        try {
            if (mat1.dims != 2 || mat2.dims != 2)
                throw std::invalid_argument("Invalid matrix dimensions in function DotProduct");
            if (mat1.shape[1] != mat2.shape[0]) {
                throw std::invalid_argument("Matrix dimensions don't match: Invalid matrix dimensions in function DotProduct");
            }
        }
        catch (const std::invalid_argument& e) {
            Logger::Error(e.what());
        }

        int row1 = mat1.shape[0], col1 = mat1.shape[1], row2 = mat2.shape[0], col2 = mat2.shape[1];
        int size = row1 * col2;

        std::vector<int> out_shape = { row1, col2 };
        std::shared_ptr<Matrix> out = std::make_shared<Matrix>(out_shape);

        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col2; j++) {
                float total = 0.0f;
                for (int k = 0; k < col1; k++) {
                    total = total + mat2.matrix_values[k * col2 + j] * mat1.matrix_values[i * col1 + k];
                }
                out->matrix_values[i * col2 + j] = total;
            }
        }

        return out;

    }

    static std::shared_ptr<Matrix> ElementwiseAddition(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, my_misc_utils::plus);
    }
    static std::shared_ptr<Matrix> ElementwiseSubtraction(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, my_misc_utils::minus);
    }
    static std::shared_ptr<Matrix> ElementwiseMultiplication(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, my_misc_utils::times);
    }
    static std::shared_ptr<Matrix> ElementwiseDivision(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, my_misc_utils::dividedby);
    }

    float Sum() {
        float sum = 0;
        for (int i = 0; i < num_vals; i++)
            sum += matrix_values[i];
        return sum;
    }

    void Map(float (*func)(float)) {
        for (int i = 0; i < num_vals; i++)
            matrix_values[i] = func(matrix_values[i]);
    }

    static std::shared_ptr<Matrix> DoubleMap(Matrix& m1, Matrix& m2, float (*func)(float, float)) {
        try {
            if (m1.num_vals != m2.num_vals) {
                throw(std::invalid_argument("Matrix shapes do not match -- Exception thrown in function Matrix::DoubleMap()"));
            }
            else {
                std::shared_ptr<Matrix> return_mat = std::make_shared<Matrix>(m1.shape);

                for (int i = 0; i < m1.num_vals; i++)
                    return_mat->matrix_values[i] = func(m1.matrix_values[i], m2.matrix_values[i]);
            }

        }
        catch(std::invalid_argument e){
            Logger::Error(e.what());
            return nullptr;
        }
    }

    static void Sigmoid(std::shared_ptr<Matrix> mat) {
        mat->Map(sigmoid_operator);
    }

    static void SigmoidPrime(std::shared_ptr<Matrix> mat) {
        mat->Map(sigmoid_prime_operator);
    }

    static void ReLU(std::shared_ptr<Matrix> mat) {
        mat->Map(relu_operator);
    }

    static void Softmax(std::shared_ptr<Matrix> mat) {
        float exp_sum = 0;
        for (int val : mat->matrix_values) {
            exp_sum += exp(val);
        }
        for (int i = 0; i < mat->num_vals; i++) {
            mat->matrix_values[i] = exp(mat->matrix_values[i]) / exp_sum;
        }
    }

    static float MeanSquareError(Matrix& targets, Matrix& outputs) {
        std::shared_ptr<Matrix> error = Matrix::ElementwiseSubtraction(targets, outputs);
        error->Square();
        float sum = error->Sum();
        return sum / targets.num_vals;
    }

    static float CategoricalCrossEntropy(Matrix& targets, Matrix& outputs) {
        std::shared_ptr<Matrix> cce_mat = DoubleMap(targets, outputs, categorical_cross_entropy_operator);
        return cce_mat->Sum();
    }

    static std::shared_ptr<Matrix> Transpose(std::shared_ptr<Matrix> mat) {

        if (mat->dims == 2) {
            std::vector<int> out_mat_shape = { mat->shape[1], mat->shape[0] }; std::shared_ptr<Matrix> out_mat = std::make_shared<Matrix>(out_mat_shape);
            for (int i = 0; i < mat->shape[0]; ++i) {
                for (int j = 0; j < mat->shape[1]; ++j) {
                    out_mat->Set({ j, i }, mat->GetVal({ i, j }));
                }
            }
            return out_mat;
        }
        else {
            Logger::Error("Cannot transpose matrix of invalid dimensions");
            return mat;
        }
    }

    static float Max(const Matrix& inp_mat) {
        float highest = 0;
        for (float val : inp_mat.matrix_values) {
            if (val > highest) {
                highest = val;
            }
        }

        return highest;
    }

    static float Average(Matrix& inp_mat) {
        float sum = inp_mat.Sum();
        return sum / inp_mat.num_vals;
    }

    static std::vector<int> NanArgmax(Matrix& inp_mat) {
        float greatest_val = 0;
        int nanargmax;
        for (int i = 0; i < inp_mat.num_vals; i++) {  
            greatest_val = (inp_mat.matrix_values[i] > greatest_val) ? inp_mat.matrix_values[i] : greatest_val;
            nanargmax = (inp_mat.matrix_values[i] > greatest_val) ? i : nanargmax;
        }

        std::vector<int> nanargmax_unraveled = UnravelIndex(nanargmax, inp_mat.shape);

        return nanargmax_unraveled;
    }

    static std::vector<int> UnravelIndex(int index, std::vector<int>& dimensions, int iteration = 0, std::vector<int> original_dimensions = {}, std::vector<int> working_answer = {}) {
        try {
            int dim_product = 1;
            for (int v : dimensions) {
                dim_product *= v;
            }

            if (index > dim_product) {
                throw std::logic_error("Index does not fit in tensor of input dimensions - error thrown in function Matrix::UnravelIndex");
            }

            int n_vals = 0;
            for (int val : dimensions) { n_vals *= val; }

            if (iteration == 0) {
                original_dimensions = dimensions;
            }
            iteration = iteration + 1;

            if (dimensions.size() == 1) {
                working_answer.push_back(n_vals);
                for (int i = working_answer.size(); i < original_dimensions.size(); i++) {
                    working_answer.push_back(0);
                }
                return working_answer;
            }
            else if (dimensions.size() > 1) {

                if (index < n_vals / 2) {
                    std::vector<int> new_dims;
                    for (int i = 0; i < dimensions.size() / 2; i++) { new_dims.push_back(dimensions[i]); }
                    UnravelIndex(index, new_dims, iteration, original_dimensions, working_answer);
                }
                else if (index > n_vals / 2) {
                    //can't just keep going pushing back into working answer, need to go by indices using original_dimensions vector
                    std::vector<int> new_dims;
                    for (int i = dimensions.size() / 2; i < dimensions.size(); i++) {
                        new_dims.push_back(dimensions[i]);
                    }
                    /*for the first half of original_dimensions{
                        working_answer.push_back(value in those indices);
                    }*/
                    index = index - n_vals / 2; //this could be a source of error
                    UnravelIndex(index, new_dims, iteration, original_dimensions, working_answer);
                }

            }
            

           

            


        }
        catch(std::logic_error e){
            Logger::Error(e.what());
        }
    }
};

#endif