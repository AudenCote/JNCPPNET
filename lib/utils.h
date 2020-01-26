std::vector<int> get_sec(std::vector<int>& v, int start, int end){
    std::vector<int> vn((end-start)+1);

    for(int i = 0; i < (end-start)+1; ++i){
        vn[i] = v[i + start];
    }

    return vn;
}

