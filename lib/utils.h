std::vector<int> get_sec(std::vector<int>& v, int start, int end){
    std::vector<int> vn((end-start)+1);

    for(int i = 0; i < (end-start)+1; ++i){
        vn[i] = v[i + start];
    }

    return vn;
}

void print_vec(std::vector<int> const input){
    for(int i = 0; i < input.size(); ++i){
        std::cout << input[i] << ' ';
    }
    std::cout << std::endl;
}