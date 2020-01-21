#include <iostream>
#include <vector>

#define LOG(x) std::cout << x << std::endl;


class Matrix {
public:
    int dims;
    std::vector<int> shape;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { }

    std::vector<int>& makeShape(std::vector<int>& ls){

        for(int i = 0; i < shape.size() - 1; ++i)
            std::vector<int> main;
            for(int j = 0; j < ls.size() - 1; ++j)
                if(j % shape[i] == 0)
                    main.push_back(ls[j:j+shape[i]])
            ls = main; //ASSIGN THIS AS REFERENCE SO THAT VECTOR IS NOT BEING COPIED?

    }
};

int main(){

    std::vector<int> params = {2, 2, 3, 4};

    Matrix mat(params);

    mat.makeShape();

    LOG(mat.dims);

    return 0;
}