
class Matrix {
private:
    static std::shared_ptr<Matrix> elementwise_operation(Matrix& mat1, Matrix& mat2, float (*func)(float, float)) {

        try {
            if (mat1.num_vals != mat2.num_vals)
                throw std::invalid_argument("Matrix dimensions do not match");
        }
        catch (const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function ElementwiseAddition" << std::endl << std::endl;
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

    void init_zeroes() {
        for (int i = 0; i < num_vals; ++i) {
            matrix_values.push_back(0.0f);
        }
    }

public:
    int dims;
    std::vector<int> shape;
    int num_vals = 1;
    std::vector<float> matrix_values;

    Matrix(std::vector<int>& params) : dims(params.size()), shape(params) {
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

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            if (dims == 0)
                return 0;

            for (int d = 0; d < dims; ++d)
                if (I[d] >= shape[d] || I[d] < 0)
                    throw std::invalid_argument("Invalid Indexing");
            if (init_list.size() != dims) {
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetChunk function instead of GetVal --");
            }

            int idx = I[0];
            for (int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);
            if (idx > num_vals)
                throw std::invalid_argument("Invalid Indexing");
            return matrix_values[idx];

        }
        catch (const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function GetVal" << std::endl << std::endl;
            return 0;
        }
    }

    std::shared_ptr<Matrix> GetChunk(std::initializer_list<int> init_list) {

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            if (dims == 0)
                return nullptr;

            for (int d = 0; d < init_list.size(); ++d) {
                if (I[d] >= shape[d] || I[d] < 0) {
                    throw std::invalid_argument("Invalid Indexing");
                }
            }

            if (init_list.size() >= dims) {
                throw std::invalid_argument("Invalid Indexing -- suggested fix: Use GetVal function instead of GetChunk --");
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
            std::cout << std::endl << e.what() << " in function GetChunk" << std::endl << std::endl;
            return nullptr;
        }

    }

    void Set(std::initializer_list<int> init_list, float val) {

        int I[init_list.size()];
        std::copy(init_list.begin(), init_list.end(), I);

        try {
            for (int d = 0; d < dims; ++d)
                if (I[d] >= shape[d] || I[d] < 0)
                    //std::cout << d << " " << I[d] << " " << shape[d] << std::endl;
                    throw std::invalid_argument("Invalid Indexing Code 0");

            int idx = I[0];
            for (int d = 1; d < dims; ++d)
                idx = idx * shape[d] + (I[d]);

            if (idx > num_vals)
                throw std::invalid_argument("Invalid Indexing Code 1");

            matrix_values[idx] = val;
        }
        catch (const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function Set" << std::endl << std::endl;
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
    }

    static std::shared_ptr<Matrix> DotProduct(Matrix& mat1, Matrix& mat2) {

        try {
            if (mat1.dims != 2 || mat2.dims != 2)
                throw std::invalid_argument("Invalid matrix dimensions");
            if (mat1.shape[1] != mat2.shape[0]) {
                throw std::invalid_argument("Matrix dimensions don't match: Invalid matrix dimensions");
            }
        }
        catch (const std::invalid_argument& e) {
            std::cout << std::endl << e.what() << " in function DotProduct" << std::endl << std::endl;
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
        return elementwise_operation(mat1, mat2, plus);
    }
    static std::shared_ptr<Matrix> ElementwiseSubtraction(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, minus);
    }
    static std::shared_ptr<Matrix> ElementwiseMultiplication(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, times);
    }
    static std::shared_ptr<Matrix> ElementwiseDivision(Matrix& mat1, Matrix& mat2) {
        return elementwise_operation(mat1, mat2, dividedby);
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

    static void Sigmoid(std::shared_ptr<Matrix> mat) {
        mat->Map(sigmoid_operator);
    }

    static void SigmoidPrime(std::shared_ptr<Matrix> mat) {
        mat->Map(sigmoid_prime_operator);
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
            std::cout << "Cannot transpose matrix of invalid dimensions" << std::endl;
            return mat;
        }
    }
};