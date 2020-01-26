#include <iostream>
#include <vector>

using namespace std;

vector<int> comb;
vector<int> nvals;


void print_vec(const vector<int>& vec) {
    static int count = 0;

    cout << "Combination no. " << (++count) << ": [ ";
    for(int i = 0; i < vec.size(); ++i) { cout << vec[i] << " "; }
    cout << "] " << endl;

}


void find_combs(int offset, int k) {
    if(k == 0){
        print_vec(comb);
        return;
    }


    for(int i = offset; i <= nvals.size() - k; ++i){
        comb.push_back(nvals[i]);
        find_combs(i + 1, k - 1);
        comb.pop_back();
    }

}



int main() {

    int n = 5;

    for(int i = 0; i < n; ++i) { nvals.push_back(i); }

    find_combs(0, 3);

    return 0;

}