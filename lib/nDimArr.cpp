#include <iostream>

using namespace std;

//The code below was written by Eric Pospischill. His Stack Overflow response in which this code was found can be 
//located at https://stackoverflow.com/questions/19883518/how-can-i-create-an-n-dimensional-array-in-c


//  Create an array with N dimensions with sizes specified in D.
int *CreateArray(size_t N, size_t D[])
{
    //  Calculate size needed.
    size_t s = sizeof(int);
    for (size_t n = 0; n < N; ++n)
        s *= D[n];

    //  Allocate space.
    return (int*) malloc(s);
}

/*  Return a pointer to an element in an N-dimensional A array with sizes
    specified in D and indices to the particular element specified in I.
*/
int *Element(int *A, size_t N, size_t D[], size_t I[])
{
    //  Handle degenerate case.
    if (N == 0)
        return A;

    //  Map N-dimensional indices to one dimension.
    int index = I[0];
    for (size_t n = 1; n < N; ++n)
        index = index * D[n] + I[n];

    //  Return address of element.
    return &A[index];
}

int main(){
	//  Create a 3*3*7*7*9 array.
	size_t Size[5] = { 3, 3, 7, 7, 9 };
	int *Array = CreateArray(5, Size);

	//  Set element [1][2][3][4][5] to -987.
	size_t arr[5] = { 1, 2, 3, 4, 5 };
	*Element(Array, 5, Size, arr ) = -987;


	size_t arr2[5] = { 1, 2, 3, 4, 5 };
	cout << *Element(Array, 5, Size, arr2 ) << endl;
}