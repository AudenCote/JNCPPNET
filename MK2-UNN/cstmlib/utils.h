namespace my_misc_utils {

    float plus(float a, float b) {
        return a + b;
    }
    float minus(float a, float b) {
        return a - b;
    }
    float times(float a, float b) {
        return a * b;
    }
    float dividedby(float a, float b) {
        return a / b;
    }

    float gen_random_float(float LO, float HI)
    {
        return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
    }


    //SO THAT INITIALIZER LISTS CAN ACTUALLY BE USED BECAUSE YOU CAN"T EVEN FUCKING INDEX THEM. FUCKING USELESS PIECES OF SHIT
    // Thank you Amir Kirsh in 2015 - https://stackoverflow.com/questions/17787394/why-doesnt-stdinitializer-list-provide-a-subscript-operator 

    template<class T>
    struct better_initializer_list {
        const std::initializer_list<T>& list;
        better_initializer_list(const std::initializer_list<T>& _list) : list(_list) {}
        T operator[](unsigned int index) {
            return *(list.begin() + index);
        }

        int size(){
            return list.size();
        }
    };

    // a function, with the short name _ (underscore) for creating 
    // the _init_list_with_square_brackets out of a "regular" std::initializer_list
    template<class T>
    better_initializer_list<T> make_useful(const std::initializer_list<T>& list) {
        return better_initializer_list<T>(list);
    }

    //USE LIKE THIS

    //void f(std::initializer_list<int> list) {
    //    cout << _(list)[2]; // subscript-like syntax for std::initializer_list!
    //}

    //int main() {
    //    f({ 1,2,3 }); // prints: 3
    //    cout << _({ 1,2,3 })[2]; // works also, prints: 3
    //    return 0;
    //}
}