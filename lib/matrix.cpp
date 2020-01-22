#include <iostream>
#include <vector>
#include "utils.h"


class Matrix {
public:
    int dims;
    std::vector<int> shape;

    Matrix (std::vector<int>& params) : dims(params.size()), shape(params) { }

    std::vector<std::vector<int>> makeShape(std::vector<int> ls){

        std::vector<std::vector<int>> main;

        for(int i = 0; i < shape.size(); ++i){
            main = {};
            for(int j = 0; j < ls.size(); ++j){
                if(j % shape[i] == 0){
                    main.push_back(get_sec(ls, j, j+shape[i] - 1));
                }
            }
            std::cout << "main: ";
            print_vec(main);
            ls = main; 
        }

        return main;

    }
};

int main(){

    std::vector<int> params = {2, 2};

    Matrix mat(params);

    print_vec(mat.makeShape({1, 2, 3, 4}));

    return 0;
}