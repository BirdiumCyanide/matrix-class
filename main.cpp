#include "Matrix.h"
#include <iostream>
using namespace std;
int main()
{
	Matrix A(5,3);
	srand(time(0));
	A.randFill(-10,10);
	cout<<A;

	Matrix U, S, V;

	A.SVD(U,S,V,1,64);
	cout<<U << S << V << U * S * transpose(V);
	system("pause");
	return 0;
}