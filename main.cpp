#include "Matrix.h"
#include <iostream>
using namespace std;
int main()
{
	double a[] = {1,2,3,4,5,6 };
	Matrix A(3, 2, a);


	Matrix U, S, V;
	A.SVD(U, S, V);
	cout<<U << S << V << U * S * transpose(V);
	
	system("pause");
	return 0;
}