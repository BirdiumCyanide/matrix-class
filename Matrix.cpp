#include "Matrix.h"

double sqr(double x)
{
	return x * x;
}
int Matrix::getIndex(int i, int j) const
{
//	if (i < 0 || i >= m_r || j < 0 || j >= m_c)
//		exit(1);
	return m_c * i + j;
}
Matrix::Matrix():m_r(0),m_c(0),m_filled(false)
{
	m_elm = 0;
}
Matrix::Matrix(int r, int c):m_r(r),m_c(c),m_filled(true)
{
	int n(r * c);
	m_elm = new double[n];
	for (int i = 0; i < n; i++)
		m_elm[i] = 0;
}
Matrix::Matrix(int r, int c, double* elm):m_c(c),m_r(r),m_filled(true)
{
	int n(r * c);
	m_elm = new double[n];
	for (int i = 0; i < n; i++)
		m_elm[i] = elm[i];
}
Matrix::Matrix(int r, int c, int x):m_r(r),m_c(c),m_filled(true)
{
	m_elm = new double[r * c];
}
Matrix::Matrix(const Matrix& mat)
{
	if (m_filled)
		delete[]m_elm;
	m_c = mat.m_c;
	m_r = mat.m_r;
	int n(m_c * m_r);
	m_elm = new double[n];
	for (int i = 0; i < n; i++)
		m_elm[i] = mat.m_elm[i];
	m_filled = true;
}
Matrix::~Matrix()
{
	if (m_filled)
		delete[]m_elm;
	m_filled = false;
}
Matrix Matrix::operator=(const Matrix& mat)
{
	if (m_c==mat.m_c&&m_r==mat.m_r)
	{
		int n(m_c * m_r);
		for (int i = 0; i < n; i++)
			m_elm[i] = mat.m_elm[i];
		return *this;
	}
	else
	{
		if (m_filled)
			delete[]m_elm;
		m_c = mat.m_c;
		m_r = mat.m_r;
		int n(m_c * m_r);
		m_elm = new double[n];
		for (int i = 0; i < n; i++)
			m_elm[i] = mat.m_elm[i];
		m_filled = true;
		return *this;
	}

}
const Matrix transpose(const Matrix& mat)
{
	Matrix tmp(mat.m_c, mat.m_r);
	int n(tmp.m_c * tmp.m_r);
	
	for (int i = 0; i < mat.m_r; i++)
		for (int j = 0; j < mat.m_c; j++)
			tmp.m_elm[tmp.getIndex(j, i)] = mat.m_elm[mat.getIndex(i, j)];
			
	return tmp;
}
double& Matrix::elm(int i, int j)
{
	return m_elm[getIndex(i, j)];
}
double Matrix::getElm(int i, int j) const
{
	return m_elm[getIndex(i, j)];
}
int Matrix::row() const
{
	return m_r;
}
int Matrix::col() const
{
	return m_c;
}
void Matrix::setElm(int i, int j, double x)
{
	m_elm[getIndex(i, j)] = x;
}

bool Matrix::isSquare()const
{
	return m_r == m_c;
}

ostream& operator<<(ostream& os,const Matrix& mat)
{
	if (mat.m_filled)
	{
		cout << mat.m_r << " * " << mat.m_c << endl;
		for (int i = 0; i < mat.m_r; i++)
		{
			for (int j = 0; j < mat.m_c; j++)
			{
				os << setw(ELM_WIDTH) << (ISZERO(mat.getElm(i, j)) ? 0 : mat.getElm(i, j));
			}
			os << endl;
		}
	}
	return os;
}
istream& operator>>(istream& is, Matrix& mat)
{
	int r, c;
	is >> r >> c;
	Matrix tmp(r, c);
	for (int i=0;i<r;i++)
		for (int j = 0; j < c; j++)
		{
			is >> tmp.m_elm[tmp.getIndex(i, j)];
		}
	mat = tmp;
	return is;
}

const Matrix Matrix::operator-() const
{
	Matrix tmp(m_r, m_c);
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
	tmp.m_elm[i] = -m_elm[i];
	return tmp;
}
const Matrix Matrix::operator+(const Matrix& mat) const
{
//	if (m_c == mat.m_c && m_r == mat.m_r)
	{
		Matrix tmp(m_r, m_c);
		int n(m_r * m_c);
		for (int i = 0; i < n; i++)
			tmp.m_elm[i] = m_elm[i] + mat.m_elm[i];
		return tmp;
	}
//	else
//		exit(1);
}
const Matrix Matrix::operator-(const Matrix& mat) const
{
//	if (m_c == mat.m_c && m_r == mat.m_r)
	{
		Matrix tmp(m_r, m_c);
		int n(m_r * m_c);
		for (int i = 0; i < n; i++)
			tmp.m_elm[i] = m_elm[i] - mat.m_elm[i];
		return tmp;
	}
//	else
//		exit(1);
}
const Matrix Matrix::operator*(const Matrix& mat) const
{
//	if (m_c == mat.m_r)
	{
		Matrix tmp(m_r, mat.m_c);
		for (int i=0;i<tmp.m_r;i++)
			for (int j = 0; j < tmp.m_c; j++)
			{
				for (int k = 0; k < m_c; k++)
				{
					tmp.m_elm[tmp.getIndex(i, j)] += m_elm[getIndex(i, k)] * mat.m_elm[mat.getIndex(k, j)];
				}
			}
		return tmp;
	}
//	else
//		exit(1);
}
const Matrix Matrix::operator*(const double x) const
{
	Matrix tmp(m_r, m_c);
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
		tmp.m_elm[i] = m_elm[i] * x;
	return tmp;
}
Matrix operator*(const double x, const Matrix& mat)
{
	Matrix tmp(mat.m_r, mat.m_c);
	int n(mat.m_r * mat.m_c);
	for (int i = 0; i < n; i++)
		tmp.m_elm[i] = x * mat.m_elm[i];
	return tmp;
}
const Matrix Matrix::operator/(const double x) const
{
//	if (!x)
//		exit(1);
	return (1 / x) * (*this);
}

Matrix& Matrix::operator+=(const Matrix& mat)
{
//	if (m_r == mat.m_r && m_c == mat.m_c)
	{
		int n(m_r * m_c);
		for (int i = 0; i < n; i++)
			m_elm[i] += mat.m_elm[i];
		return *this;
	}
//	else
//		exit(1);
}
Matrix& Matrix::operator-=(const Matrix& mat)
{
//	if (m_r == mat.m_r && m_c == mat.m_c)
	{
		int n(m_r * m_c);
		for (int i = 0; i < n; i++)
			m_elm[i] -= mat.m_elm[i];
		return *this;
	}
//	else
//		exit(1);
}
Matrix& Matrix::operator*=(const double x)
{
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
		m_elm[i] *= x;
	return *this;
}
Matrix& Matrix::operator/=(const double x)
{
//	if (ISZERO(x))
//		exit(1);
	double t = 1.0 / x;
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
		m_elm[i] *= t;
	return *this;
}

bool Matrix::operator==(const Matrix& mat) const
{
	if (m_r != mat.m_r || m_c != mat.m_c)
		return false;
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
		if (m_elm[i] != mat.m_elm[i])
			return false;
	return true;
}
bool Matrix::operator!=(const Matrix& mat) const
{
	return !((*this) == mat);
}

void Matrix::multRow(int r, double k)
{
	for (int i = 0; i < m_c; i++)
		m_elm[getIndex(r, i)] *= k;
}
void Matrix::multCol(int c,double k)
{
	for (int i = 0; i < m_r; i++)
		m_elm[getIndex(i, c)] *= k;
}
void Matrix::swapRow(int r1, int r2)
{
	for (int i = 0; i < m_c; i++)
		swap(m_elm[getIndex(r1, i)], m_elm[getIndex(r2, i)]);
}
void Matrix::swapCol(int c1, int c2)
{
	for (int i = 0; i < m_r; i++)
		swap(m_elm[getIndex(i, c1)], m_elm[getIndex(i, c2)]);
}
void Matrix::addRow(int r0, int r, double k = 1.0, int l = 0)
{
	for (int i = l; i < m_c; i++)
	{
		m_elm[getIndex(r0, i)] += m_elm[getIndex(r, i)] * k;
	}
}
void Matrix::addCol(int c0, int c, double k=1.0)
{
	for (int i = 0; i < m_r; i++)
	{
		m_elm[getIndex(i, c0)] += m_elm[getIndex(i, c)] * k;
	}
}

double Matrix::det()const
{
	if (!isSquare())
		exit(1);
	Matrix tmp(*this);
	double max;
	int maxRow;
	double k = 1;
	for (int i = 0; i < m_r; i++)
	{
		max = fabs(tmp.m_elm[getIndex(i, i)]);
		maxRow = i;
		for (int j = i + 1; j < m_r; j++)
		{
			if (fabs(tmp.m_elm[getIndex(j, i)]) > max)
			{
				max = fabs(tmp.m_elm[getIndex(j, i)]);
				maxRow = j;
			}
		}
		if (ISZERO(max)) return 0;
		if (maxRow != i)
		{
			tmp.swapRow(i, maxRow);
			k *= -1;
		}
		for (int j = i + 1; j < m_r; j++)
		{
			tmp.addRow(j, i, -tmp.m_elm[getIndex(j, i)] / tmp.m_elm[getIndex(i, i)]);
		}
	}
	for (int i = 0; i < m_r; i++)
	{
		k *= tmp.m_elm[getIndex(i, i)];
	}
	return k;
}
bool Matrix::LUfact(Matrix& L, Matrix& U)const
{
	Matrix tmpU(*this);
	Matrix tmpL = identityMat(m_r);
	for (int i = 0; i < m_r; i++)
	{
		for (int j = i + 1; j < m_r; j++)
		{
			if (ISZERO(tmpU.m_elm[getIndex(i, i)]))
				return false;
			double k = tmpU.m_elm[getIndex(j, i)] / tmpU.m_elm[getIndex(i, i)];
			tmpU.addRow(j, i, -k);
			tmpL.setElm(j, i, k);
		}
	}
	L = tmpL;
	U = tmpU;
	return true;
}
bool Matrix::PALUfact(Matrix& P, Matrix& L, Matrix& U)const
{
	Matrix tmpU(*this);
	Matrix tmpP = identityMat(m_r);
	Matrix tmpL(tmpP);

	double max;
	int maxRow;

	for (int i = 0; i < m_r; i++)
	{
		max = fabs(tmpU.m_elm[getIndex(i, i)]);
		maxRow = i;
		for (int j = i + 1; j < m_r; j++)
		{
			if (fabs(tmpU.m_elm[getIndex(j, i)]) > max)
			{
				max = fabs(tmpU.m_elm[getIndex(j, i)]);
				maxRow = j;
			}
		}
		if (ISZERO(max))
		{
			continue;
		}

		if (maxRow != i)
		{
			tmpU.swapRow(i, maxRow);
			tmpP.swapRow(i, maxRow);
		}
		for (int j = i + 1; j < m_r; j++)
		{
			double k = tmpU.elm(j,i) / tmpU.elm(i, i);
			tmpU.addRow(j, i, -k, i);
			tmpU.setElm(j, i, k);
		}
	}
	for (int i = 0; i < m_r - 1; i++)
	{
		for (int j = i + 1; j < m_r; j++)
		{
			tmpL.setElm(j, i, tmpU.m_elm[getIndex(j, i)]);
			tmpU.setElm(j, i, 0);
		}
	}
	U = tmpU;
	P = tmpP;
	L = tmpL;
	return true;
}
bool Matrix::solveWithPALU(Matrix& P, Matrix& L, Matrix& U, Matrix& B, Matrix& Sol)const
{
//	if (B.m_c != 1)
//		return false;
	Matrix tmpB = P * B;
	Matrix C(tmpB);
	for (int i = 0; i < m_r; i++)
	{
		for (int j = 0; j < i ; j++)
		{
			C.m_elm[i] -= L.elm(i, j) * C.m_elm[j];
		}
	}
	Matrix D(C.m_r, C.m_c, 0);
	for (int i = 0; i < m_r; i++)
	{
		D.m_elm[i] = 1;
	}
	for (int i = m_r - 1; i > -1; i--)
	{
		int j = 0;
		while (ISZERO(U.elm(i, j))&&(++j)<m_c);

		if (j == m_c)
		{
			if (ISZERO(C.m_elm[i]))
				continue;
			else
				return false;
		}
		D.m_elm[j] = C.m_elm[i];
		for (int k = j+1; k < m_c; k++)
		{
			D.m_elm[j] -= U.elm(i, k) * D.m_elm[k];
		}
		D.m_elm[j] /= U.elm(i, j);
	}
	Sol = D;
	return true;
}
bool Matrix::solveWithPALU(Matrix& B, Matrix& Sol) const
{
	Matrix P, L, U;
	if (!PALUfact(P, L, U))
		return false;
	if (!solveWithPALU(P, L, U, B, Sol))
		return false;

	return true;
}
bool Matrix::solSet(Matrix& Sol) const
{
	Matrix P, L, U;
	PALUfact(P, L, U);

	int n(m_r);
	Matrix freeVar(n, 1), majorElm(n, 1, 0);

	for (int i = 0; i <n; i++)
	{
		int j = 0;
		while (j < n && ISZERO(U.elm(i, j)))j++;
		majorElm.m_elm[i] = j;
		if (j < n)
		{
			freeVar.m_elm[j] = 1;
		}
	}

	for (int i = 0; i < n; i++)
	{
		int j = i + 1;
		while (majorElm.m_elm[j] == majorElm.m_elm[i]&&majorElm.m_elm[j]<n)
		{
			U.addRow(j, i, -U.elm(j, majorElm.m_elm[i]) / U.elm(i, majorElm.m_elm[j]));
			j++;
		}
		for (int k = i; k < n; k++)
		{
			j = 0;
			while (j < n && ISZERO(U.elm(k, j)))j++;
			majorElm.m_elm[k] = j;
		}
	}

	int rank(0);
	for (int i = 0; i < n; i++)
	{
		int j = 0;
		while (j < n && ISZERO(U.elm(i, j)))j++;
		majorElm.m_elm[i] = j;
		if (j < n)
			rank++;
	}
	
	if (rank == n)
	{
		Matrix S(n, 1);
		Sol = S;
	}
	else
	{
		Matrix D(n, n - rank);
		int l(0);
		for (int k = 0; k < n; k++)
		{
			if (!freeVar.m_elm[k])
			{
				D.elm(k, l) = 1;
				for (int i = n - 1; i > -1; i--)
				{
					int j = 0;
					while (j < n && ISZERO(U.elm(i, j)))j++;
					if (j < n)
					{
						for (int p = j + 1; p < n; p++)
						{
							D.elm(j, l) -= U.elm(i, p) * D.elm(p, l);
						}
						D.elm(j, l) /= U.elm(i, j);
					}
				}
				l++;
			}
		}
		Sol = D;
	}
	return true;
}

bool Matrix::inv(Matrix& INV)const
{
	if (!isSquare())
		return false;
	
	Matrix P, L, U;
	if (PALUfact(P, L, U))
	{
		int n = m_r;
		for (int i = 0; i < n; i++)
		{
			if (ISZERO(U.elm(i, i)))
				return false;
		}
		Matrix tmpINV(n, n);
		for (int i = 0; i < n; i++)
		{
			Matrix B = unitColVec(n, i), S;
			solveWithPALU(P, L, U, B, S);
			for (int j = 0; j < n; j++)
			{
				tmpINV.setElm(j, i, S.m_elm[j]);
			}
		}
		INV = tmpINV;
		return true;
	}
	return false;
}
bool Matrix::solveWithINV(Matrix& B, Matrix& Sol) const
{
	Matrix INV;
	if (!inv(INV))
		return false;
	Sol = INV * B;
	return true;
}
double Matrix::MatrixNorm()const
{
	double max = 0;
	for (int i = 0; i < m_r; i++)
	{
		double s = 0;
		for (int j = 0; j < m_c; j++)
		{
			s += fabs(getElm(i, j));
		}
		if (s > max)
			max = s;
	}
	return max;
}
bool Matrix::JacobiIteration(Matrix& B, Matrix& X, double accu, int maxStep)const
{
	if (!isSquare())
		return false;
	int n = m_r;
	int step = 0;
	Matrix C(n, 1);
	Matrix R = B - (*this) * C;
	while (step < maxStep && R.MatrixNorm()>=accu)
	{
		Matrix T(B);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i != j)
				{
					T.m_elm[i] -= C.m_elm[j] * getElm(i, j);
				}
			}
			if (ISZERO(getElm(i, i)))
				return false;
			else
				T.m_elm[i] /= getElm(i, i);
		}
		C = T;
		R = B - (*this) * C;
		if (R.MatrixNorm() < accu)
		{
			X = C;
			return true;
		}
		step++;
	}
	return false;
}
bool Matrix::SOR(Matrix& B, Matrix& X, double w, double accu, int maxStep)const
{
	if (!isSquare())
		return false;
	int n = m_r;
	int step = 0;
	Matrix C(n, 1);
	Matrix R = B - (*this) * C;
	while (step < maxStep && R.MatrixNorm() >= accu)
	{
		Matrix T(B);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i < j)
				{
					T.m_elm[i] -= C.m_elm[j] * getElm(i, j);
				}
				if (i > j)
				{
					T.m_elm[i] -= T.m_elm[j] * getElm(i, j);
				}
			}
			if (ISZERO(getElm(i, i)))
				return false;
			else
				T.m_elm[i] = T.m_elm[i] / getElm(i, i) * w + (1 - w) * C.m_elm[i];

		}
		C = T;
		R = B - (*this) * C;
		if (R.MatrixNorm() < accu)
		{
			X = C;
			return true;
		}
		step++;
	}
	return false;
}
bool Matrix::GaussSeidel(Matrix& B, Matrix& X, double accu, int maxStep)const
{
	if (!isSquare())
		return false;
	int n = m_r;
	int step = 0;
	Matrix C(n, 1);
	Matrix R = B - (*this) * C;
	while (step < maxStep && R.MatrixNorm() >= accu)
	{
		Matrix T(B);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i < j)
				{
					T.m_elm[i] -= C.m_elm[j] * getElm(i, j);
				}
				if (i > j)
				{
					T.m_elm[i] -= T.m_elm[j] * getElm(i, j);
				}
			}
			if (ISZERO(getElm(i, i)))
				return false;
			else
				T.m_elm[i] = T.m_elm[i] / getElm(i, i);

		}
		C = T;
		R = B - (*this) * C;
		if (R.MatrixNorm() < accu)
		{
			X = C;
			return true;
		}
		step++;
	}
	return false;
}
bool Matrix::ConjugateGradient(Matrix& B, Matrix& X)
{
	if (isSquare())
	{
		int n(m_r);
		Matrix x0(n, 1);
		Matrix r = B;
		Matrix d(r);
		double t, alpha, beta;
		for (int i = 0; i < n; i++)
		{
			if (ISZERO(r.MatrixNorm()))
			{
				X = x0;
				return true;
			}
			t = innerProduct(r, r);
			alpha = t / innerProductWithMetricMat(d, *this, d);
			x0 += alpha * d;
			r -= alpha * (*this) * d;
			beta = innerProduct(r, r) / t;
			d = r + beta * d;
		}
		X = x0;
		return true;
	}
	else
		return false;
}

double Matrix::cond()const
{
	Matrix INV;
	if (inv(INV))
	{
		return MatrixNorm() * INV.MatrixNorm();
	}
	else
		return 0;
}

const Matrix Matrix::getCol(int i)const
{
	if (i < 0 || i >= m_c)
	{
		exit(1);
	}
	else
	{
		Matrix tmp(m_r, 1);
		for (int j = 0; j < m_r; j++)
		{
			tmp.m_elm[j] = getElm(j, i);
		}
		return tmp;
	}
}
const Matrix Matrix::getRow(int i)const
{
	if (i < 0 || i >= m_r)
	{
		exit(1);
	}
	else
	{
		Matrix tmp(1, m_c);
		for (int j = 0; j < m_r; j++)
		{
			tmp.m_elm[j] = getElm(i, j);
		}
		return tmp;
	}
}

void Matrix::fillWith(double x)
{
	int n(m_r*m_c);
	for (int i = 0; i < n; i++)
		m_elm[i] = x;
}
void Matrix::randFill(double l, double r)
{
	int n(m_r * m_c);
	for (int i = 0; i < n; i++)
	{
		m_elm[i] = l + ((double)rand()) / RAND_MAX * (r - l);
	}
}
void Matrix::symmetrize(int x)
{
	if (isSquare())
	{
		int n(m_r);
		if (x)
		{
			for (int i = 0; i < n - 1; i++)
				for (int j = i + 1; j < n; j++)
					elm(i, j) = elm(j, i);
		}
		else
		{
			for (int i = 1; i < n; i++)
				for (int j = 0; j < i; j++)
					elm(i, j) = elm(j, i);
		}
	}
	else
		return;
}

bool Matrix::Cholesky(Matrix& R) const
{
	if (isSquare())
	{
		int n(m_r);
		R = *this;
		for (int i = 0; i < n; i++)
		{
			if (R.elm(i, i) <= 0)
				return false;
			R.elm(i, i) = sqrt(R.elm(i, i));
			for (int j = i + 1; j < n; j++)
				R.elm(i, j) /= R.elm(i, i);
			for (int j = i + 1; j < n; j++)
			{
				for (int k = j ; k < n; k++)
				{
					R.elm(j, k) -= R.elm(i, j) * R.elm(i, k);
				}
			}
		}
		for (int i = 1; i < n; i++)
			for (int j = 0; j < i; j++)
				R.elm(i, j) = 0;
		return true;
	}
	else
		return false;
}

const Matrix identityMat(int n)
{
	Matrix tmp(n, n);
	for (int i = 0; i < n; i++)
		tmp.setElm(i, i, 1.0);
	return tmp;
}
const Matrix scalarMat(int n, double k)
{
	Matrix tmp(n, n);
	for (int i = 0; i < n; i++)
		tmp.setElm(i, i, k);
	return tmp;
}
const Matrix unitColVec(int n, int i)
{
	Matrix tmp(n, 1);
	tmp.setElm(i, 0, 1);
	return tmp;
}
const Matrix hilbertMat(int n)
{
	Matrix tmp(n, n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			tmp.setElm(i, j, 1.0 / (i + j + 1.0));
	return tmp;
}
const Matrix diagMat(int n, double* d)
{
	Matrix tmp(n, n);
	for (int i = 0; i < n; i++)
		tmp.setElm(i, i, d[i]);
	return tmp;
}
const Matrix minorDiagMat(int n, double* d)
{
	Matrix tmp(n, n);
	for (int i = 0; i < n; i++)
		tmp.elm(i, n - i - 1) = d[i];
	return tmp;
}

double innerProduct(const Matrix& A, const Matrix& B)
{
	double s(0);
	for (int i = 0; i < A.m_r; i++)
		s += A.m_elm[i] * B.m_elm[i];
	return s;
}
double innerProductWithMetricMat(const Matrix& A, const Matrix& G, const Matrix& B)
{
	Matrix t = transpose(A) * G * B;
	return t.m_elm[0];
}
const Matrix normalize(const Matrix& A, int i = 0)
{
	double s = 0;
	for (int j = 0; j < A.m_r; j++)
	{
		s += sqr(A.getElm(j, i));
	}
	s = sqrt(s);
	if (ISZERO(s))
		return A;
	Matrix tmp(A);
	s = 1.0 / s;
	for (int j = 0; j < A.m_r; j++)
	{
		tmp.elm(j, i) *= s;
	}
	return tmp;
}
void Matrix::normalizeCol(int i)
{
	double s = 0;
	for (int j = 0; j < m_r; j++)
	{
		s += sqr(elm(j, i));
	}
	s = sqrt(s);
	if (!ISZERO(s))
	{
		s = 1.0 / s;
		for (int j = 0; j < m_r; j++)
		{
			elm(j, i) *= s;
		}
	}
}

bool Matrix::GramSchmidt()
{
	Matrix Q(m_r, m_c),y,Qi;
	for (int j = 0; j < m_c; j++)
	{
		y = getCol(j);

		for (int i = 0; i < j; i++)
		{
			Qi = Q.getCol(i);
			y -= innerProduct(Qi, y) * Qi;
		}

		double s = innerProduct(y, y);
		if (ISZERO(s))
			return false;
		s = 1.0 / sqrt(s);
		for (int i = 0; i < m_r; i++)
		{
			Q.elm(i, j) = y.m_elm[i] * s;
		}
		
	}
	*this = Q;
	return true;
}
bool Matrix::QRfact(Matrix& Q, Matrix& R) const
{
	Q = *this;
	Matrix tmpR(m_c, m_c), y, Qi;
	for (int j = 0; j < m_c; j++)
	{
		y = getCol(j);

		for (int i = 0; i < j; i++)
		{
			Qi = Q.getCol(i);
			tmpR.elm(i, j) = innerProduct(Qi, y);
			y -= tmpR.elm(i, j) * Qi;
		}

		double s = innerProduct(y, y);
		if (ISZERO(s))
			return false;
		s =  sqrt(s);
		
		tmpR.elm(j, j) = s;
		for (int i = 0; i < m_r; i++)
		{
			Q.elm(i, j) = y.m_elm[i] / s;
		}
	}
	R = tmpR;
	return true;
}
const Matrix Householder(const Matrix& v)
{
	return identityMat(v.row()) - 2 * v * transpose(v) / innerProduct(v, v);
}
const Matrix Householder(const Matrix& x, const Matrix& w)
{
	return Householder(w - x);
}
bool Matrix::QRfactWithHouseholder(Matrix& Q, Matrix& R) const
{
	Matrix I = identityMat(m_r), H, Hj, v1, v2;
	Q = I;
	R = *this;
	for (int j = 0; j < m_c; j++)
	{
		Matrix x(m_r - j, 1, 0), w(m_r - j, 1);
		double s = 0;
		for (int i = j; i < m_r; i++)
		{
			s += sqr(R.elm(i, j));
			x.m_elm[i - j] = R.elm(i, j);
		}
		w.m_elm[0] = sqrt(s);
		v1 = w - x;
		v2 = w + x;
		double x1 = innerProduct(v1, v1);
		double x2 = innerProduct(v2, v2);
		if (x1 < x2)
		{
			if (ISZERO(sqrt(x2)))
				return false;
			w.m_elm[0] = -w.m_elm[0];
			Hj = Householder(v2);
		}
		else
		{
			if (ISZERO(sqrt(x1)))
				return false;
			Hj = Householder(v1);
		}
		H = I;
		for (int i = j; i < m_r; i++)
		{
			for (int k = j; k < m_r; k++)
			{
				H.elm(i, k) = Hj.elm(i - j, k - j);
			}
		}
		Q = Q * H;
		R = H * R;
		for (int i = 0; i < R.m_r && i < R.m_c; i++)
		{
			if (R.elm(i, i) < 0)
			{
				R.multRow(i, -1);
				Q.multCol(i, -1);
			}
		}
	}
	return true;
}

bool Matrix::powerIteration(double& eigVal, Matrix& eigVec, int maxStep)
{
	if (isSquare())
	{
		Matrix X = unitColVec(m_r, 0), U;
		for (int i = 0; i < maxStep; i++)
		{
			U = normalize(X);
			X = (*this) * U;
		}
		eigVal = innerProductWithMetricMat(U, *this, U);
		eigVec = normalize(U);
		return true;
	}
	else
	{
		return false;
	}
}
bool Matrix::powerIteration(double& eigVal, Matrix& eigVec,const Matrix& x0,int maxStep)
{
	if (isSquare())
	{
		Matrix X = x0, U;
		for (int i = 0; i < maxStep; i++)
		{
			U = normalize(X);
			X = (*this) * U;
		}
		eigVal = innerProductWithMetricMat(U, *this, U);
		eigVec = normalize(U);
		return true;
	}
	else
	{
		return false;
	}
}

void Matrix::toUpperHessenberg(Matrix& Q, Matrix& B) const
{
	B = *this;
	int n = m_c;
	Q = identityMat(n);
	Matrix v1, v2, H, Hj, I(Q);
	for (int j = 0; j < n - 1; j++)
	{
		Matrix x(n - j - 1, 1, 0), w(n - j - 1, 1);
		double s = 0;
		for (int i = j + 1; i < n; i++)
		{
			s += sqr(B.elm(i, j));
			x.m_elm[i - j - 1] = B.elm(i, j);
		}
		w.m_elm[0] = sqrt(s);
		v1 = w - x;
		v2 = w + x;
		double x1 = innerProduct(v1, v1);
		double x2 = innerProduct(v2, v2);
		if ((!ISZERO(x1)) || (!ISZERO(x2)))
		{
			if (x1 < x2)
			{
				w.m_elm[0] = -w.m_elm[0];
				Hj = Householder(v2);
			}
			else
			{
				Hj = Householder(v1);
			}
			H = I;
			for (int i = j + 1; i < n; i++)
			{
				for (int k = j + 1; k < n; k++)
				{
					H.elm(i, k) = Hj.elm(i - j - 1, k - j - 1);
				}
			}
			Q = H * Q;
			B = H * B * transpose(H);
		}
	}
}
void Matrix::shrink(int x)
{
	int n = m_r;
	Matrix tmp(n-x, n-x, 0);
	for (int i = 0; i < n-x; i++)
		for (int j = 0; j < n-x; j++)
			tmp.elm(i, j) = elm(i, j);
	*this = tmp;
}
void Matrix::QRmethod(Matrix& LAM, double accu, double maxStep) const
{
	int n(m_r);
	int count;
	double max, x, delta, det, tr;
	Matrix A(*this);
	Matrix Q, R, S;
	Matrix tmpLAM(n, 2);
	while (n > 1)
	{
		count = 0;
		while (count++ < maxStep)
		{
			max = 0;
			for (int i = 0; i < n - 1; i++)
			{
				x = fabs(A.elm(n - 1, i));
				if (x > max)
					max = x;
			}
			if (max < accu) break;
			S = scalarMat(n, A.elm(n - 1, n - 1));
			(A - S).QRfactWithHouseholder(Q, R);
			A = R * Q + S;
		}

		if (count < maxStep)
		{
			tmpLAM.elm(n-1, 0) = A.elm(n-1, n-1);
			n--;
			A.shrink(1);
		}
		else
		{
			tr = A.elm(n - 1, n - 1) + A.elm(n - 2, n - 2);
			det = A.elm(n - 2, n - 2) * A.elm(n - 1, n - 1) - A.elm(n - 2, n - 1) * A.elm(n - 1, n - 2);
			delta = sqr(tr) - 4 * det;
			if (delta < 0)
			{
				tmpLAM.elm(n - 2, 0) = 0.5 * tr;
				tmpLAM.elm(n - 1, 0) = 0.5 * tr;
				tmpLAM.elm(n - 2, 1) = 0.5 * sqrt(-delta);
				tmpLAM.elm(n - 1, 1) = -0.5 * sqrt(-delta);
			}
			else
			{
				tmpLAM.elm(n - 2, 0) = 0.5 * (tr + sqrt(delta));
				tmpLAM.elm(n - 1, 0) = 0.5 * (tr - sqrt(delta));
			}
			n -= 2;
			A.shrink(2);
		}
	}
	if (n > 0)
		tmpLAM.elm(0, 0) = A.elm(0, 0);
	LAM = tmpLAM;
}
void Matrix::symEig(Matrix& LAM, Matrix& Q, double accu, double maxStep) const
{
	Matrix P = identityMat(m_r), R = *this, T;
	Q = P;
	for (int i = 0; i < maxStep; i++)
	{
		(R * P).QRfactWithHouseholder(P, R);
		Q = Q * P;
	}
	T = R * P;
	Matrix L(m_r, 1, 0);
	for (int i = 0; i < m_r; i++)
		L.m_elm[i] = T.elm(i, i);
	LAM = L;
}
void Matrix::eig(Matrix& LAM, double accu, double maxStep) const
{
	Matrix Q, B;
	toUpperHessenberg(Q, B);
	B.QRmethod(LAM, accu, maxStep);
}
void Matrix::eigVec(double lam, Matrix& V, int x) const
{
	Matrix L = scalarMat(m_r, lam);
	(L - *this).solSet(V);
	if (x)
	for (int i = 0; i < V.m_c; i++)
	{
		V.normalizeCol(i);
	}
}
void Matrix::SVD(Matrix& U, Matrix& S, Matrix& V, double accu, double maxStep) const
{
	Matrix LAM;
	(transpose(*this) * (*this)).symEig(LAM, V, accu, maxStep);

	int l(0), r(LAM.m_r - 1);
	while (l < r)
	{
		while (!ISZERO(LAM.m_elm[l++]));
		l--;
		while (ISZERO(LAM.m_elm[r--]));
		r++;
		if (l < r)
		{
			swap(LAM.m_elm[l], LAM.m_elm[r]);
			V.swapCol(l, r);
		}
	}

	Matrix tmpS(m_r, m_c);
	for (int i = 0; i < LAM.m_r; i++)
	{
		tmpS.elm(i, i) = sqrt(LAM.m_elm[i]);
	}
	S = tmpS;
	U = (*this) * V;
	for (int i = 0; i < LAM.m_r; i++)
	{
		if (!ISZERO(S.elm(i, i)))
		{
			U.multCol(i, 1.0 / S.elm(i, i));
		}
	}

	Matrix Q, R;
	U.QRfactWithHouseholder(Q, R);
	U = Q;
}
const Matrix Matrix::subMat(int u, int d, int l, int r) const
{
	Matrix tmp(d - u + 1, r - l + 1, 0);
	for (int i = u; i <= d; i++)
		for (int j = l; j <= r; j++)
			tmp.elm(i - u, j - l) = getElm(i, j);
	return tmp;
}

const Matrix colCombine(const Matrix& A, const Matrix& B)
{
	Matrix tmp(A.m_r, A.m_c + B.m_c, 0);
	for (int i = 0; i < A.m_r; i++)
	{
		for (int j = 0; j < A.m_c; j++)
		{
			tmp.elm(i, j) = A.getElm(i, j);
		}
		for (int j = 0; j < B.m_c; j++)
		{
			tmp.elm(i, j + A.m_c) = B.getElm(i, j);
		}
	}
	return tmp;
}