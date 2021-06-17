#pragma once
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#define ELM_PRECISION 6
#define ELM_WIDTH 12
#define EPSILON 1e-12
#define ACCURACY 1e-12
#define MAX_STEP 128
#define ISZERO(X) abs(X)<EPSILON 
using namespace std;
class Matrix
{
protected:
	int m_c;
	int m_r;
	double* m_elm;
	bool m_filled = false;
	int getIndex(int i, int j) const;
public:
	Matrix();
	Matrix(int r, int c);
	Matrix(int r, int c, double* elm);//initialize with array
	Matrix(int r, int c, int x);//do not fill with zero
	Matrix(const Matrix& mat);
	~Matrix();
	
	double& elm(int i, int j);//for assigning value, cannot be called by const object
	void setElm(int i, int j, double x);//also for assigning value
	double getElm(int i, int j) const;//get the value
	int row() const;//get row number
	int col() const;//get column number

	bool isSquare()const;

	friend const Matrix transpose(const Matrix& mat);

	//operator overload
	Matrix operator=(const Matrix& mat);
	friend ostream& operator<<(ostream& os, const Matrix& mat);
	friend istream& operator>>(istream& is, Matrix& mat);

	const Matrix operator-() const;

	const Matrix operator+(const Matrix& mat) const;
	const Matrix operator-(const Matrix& mat) const;
	const Matrix operator*(const Matrix& mat) const;
	const Matrix operator*(const double x) const;
	friend Matrix operator*(const double x, const Matrix& mat);
	const Matrix operator/(const double x) const;
	
	Matrix& operator+=(const Matrix& mat);
	Matrix& operator-=(const Matrix& mat);
	Matrix& operator*=(const double x);
	Matrix& operator/=(const double x);

	bool operator==(const Matrix& mat) const;
	bool operator!=(const Matrix& mat) const;

	double det()const;

	//solve AX=B, A is reversible, B is a column vector 
	void multRow(int r, double k);
	void multCol(int c, double k);
	void swapRow(int r1, int r2);
	void swapCol(int c1, int c2);
	void addRow(int r0,int r, double k, int l);
	void addCol(int c0,int c, double k);

	bool LUfact(Matrix& L, Matrix& U) const;
	bool PALUfact(Matrix& P, Matrix& L, Matrix& U) const;
	bool solveWithPALU(Matrix& P, Matrix& L, Matrix& U, Matrix& b, Matrix& Sol) const;
	bool solveWithPALU(Matrix& B, Matrix& Sol) const;
	bool solSet(Matrix& Sol) const;

	bool inv(Matrix& INV) const;
	bool solveWithINV(Matrix& B, Matrix& Sol) const;

	double MatrixNorm() const;
	bool JacobiIteration(Matrix& B, Matrix& X, double accu = ACCURACY, int maxStep = MAX_STEP) const;
	bool SOR(Matrix& B, Matrix& X, double w = 1.25, double accu = ACCURACY, int maxStep = MAX_STEP) const;
	bool GaussSeidel(Matrix& B, Matrix& X, double accu = ACCURACY, int maxStep = MAX_STEP) const;
	//solve AX=B, A is symmetric
	bool ConjugateGradient(Matrix& B,Matrix& X);

	double cond() const;

	const Matrix getCol(int i) const;
	const Matrix getRow(int i) const;
	const Matrix subMat(int u, int d, int l, int r) const;

	void fillWith(double x);
	void randFill(double l = 0.0, double r = 1.0);
	void symmetrize(int x = 0);

	bool Cholesky(Matrix& R) const;

	friend double innerProduct(const Matrix& A, const Matrix& B);
	friend double innerProductWithMetricMat(const Matrix& A, const Matrix& G, const Matrix& B);
	friend const Matrix normalize(const Matrix& A, int i);
	void normalizeCol(int i = 0);
	bool GramSchmidt();
	bool QRfact(Matrix& Q, Matrix& R) const;
	bool QRfactWithHouseholder(Matrix& Q, Matrix& R) const;

	//for eigenvalue
	bool powerIteration(double& eigVal, Matrix& eigVec, int maxStep = MAX_STEP);
	bool powerIteration(double& eigVal, Matrix& eigVec, const Matrix& x0, int maxStep = MAX_STEP);

	void toUpperHessenberg(Matrix& Q, Matrix& B) const;
	void shrink(int x);
	void QRmethod(Matrix& LAM, double accu = ACCURACY, double maxStep = MAX_STEP) const;
	void symEig(Matrix& LAM, Matrix& Q, double accu = ACCURACY, double maxStep = MAX_STEP) const;
	void eig(Matrix& LAM, double accu = ACCURACY, double maxStep = MAX_STEP) const;
	void eigVec(double lam, Matrix& V, int x = 0) const;

	void SVD(Matrix& U, Matrix& S, Matrix& V, double accu = ACCURACY, double maxStep = MAX_STEP) const;
	void symSVD(Matrix& U, Matrix& S, Matrix& V, double accu = ACCURACY, double maxStep = MAX_STEP) const;
	friend const Matrix colCombine(const Matrix& A, const Matrix& B);
};

const Matrix identityMat(int n);
const Matrix scalarMat(int n, double k);
const Matrix unitColVec(int n, int i);
const Matrix hilbertMat(int n);
const Matrix diagMat(int n, double* d);
const Matrix minorDiagMat(int n, double* d);
const Matrix Householder(const Matrix& v);
const Matrix Householder(const Matrix& x, const Matrix& w);
double sqr(double x);

