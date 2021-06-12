#include "Matrix.h"
#include <iostream>
using namespace std;
int main()
{
	double a[] = {0,-0.5,3,0,0,0 };
	Matrix A(3, 2, a);


	Matrix U, S, V;
	A.SVD(U, S, V);
	cout<<U << S << V;
	
	return 0;
}