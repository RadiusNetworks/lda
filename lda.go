// Package lda provides methods for calculating linear discriminant analysis (LDA).
// LDA can be used as a dimensionality reduction technique and as a classifier.
// Both capabilities are often used in the realm of machine learning and statistical modeling. This package
// provides a prediction method that can classify input data based on previous calculations and
// feature extraction from training data.
package lda

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// LD is a type for computing and extracting the linear discriminant analysis of a
// matrix. The results of the linear discriminant analysis are only valid
// if the call to LinearDiscriminant was successful.
type LD struct {
	n, p  int        // n = # of rows, p = # of columns
	k     int        // number of classes
	ct    []float64  // Constant term of discriminant function of each class
	mu    *mat.Dense // Mean vectors of each class
	svd   *mat.SVD
	ok    bool
	eigen mat.Eigen //Eigen values of common variance matrix
}

// LinearDiscriminant performs linear discriminant analysis on the
// matrix of the input data, which is represented as an n×p matrix x,
// where each row is an observation and each column is a variable.
//
//
// Parameter x is a matrix of input/training data.
// Parameter y is an array of input/training labels in [0,k)
// where k is the number of classes.
// Returns true iff the analysis was successful.
func (ld *LD) LinearDiscriminant(x mat.Matrix, y []int) (err error) {
	ld.n, ld.p = x.Dims()
	if y != nil && len(y) != ld.n {
		return fmt.Errorf("The sizes of X and Y don't match")
	}
	var labels []int
	var labelMap = map[int]int{}
	for _, label := range y {
		if labelMap[label] == 0 {
			labelMap[label] = 1
			labels = append(labels, label)
		} else {
			labelMap[label]++
		}
	}

	// Create a new array with labels and go through the array of y values and if
	// it doesn't exist then add it to the new array
	sort.Ints(labels)

	if len(labels) == 0 {
		return fmt.Errorf("No data to analyze")
	}
	if labels[0] != 0 {
		return fmt.Errorf("Label does not start from zero")
	}
	for i := 0; i < len(labels); i++ {
		if labels[i] < 0 {
			return fmt.Errorf("Negative class label")
		}
		if i > 0 && labels[i]-labels[i-1] > 1 {
			return fmt.Errorf("Missing class")
		}
	}

	// Tol is a tolerence to decide if a covariance matrix is singular (det is zero)
	// Tol will reject variables whose variance is less than tol
	var tol = 1e-4

	ld.k = len(labels)
	if ld.k < 2 {
		return fmt.Errorf("Only one class")
	}
	if tol < 0.0 {
		return fmt.Errorf("Invalid tol")
	}
	if ld.n <= ld.k {
		return fmt.Errorf("Sample size is too small")
	}

	// Number of instances in each class
	ni := make([]int, ld.k)

	// Common mean vector
	var colmean []float64
	for i := 0; i < ld.p; i++ {
		var col = mat.Col(nil, i, x)
		var sum float64
		for _, value := range col {
			sum += value
		}
		colmean = append(colmean, sum/float64(ld.n))
	}

	// Class mean vectors
	// mu is a k x ld.p matrix
	ld.mu = mat.NewDense(ld.k, ld.p, make([]float64, ld.k*ld.p, ld.k*ld.p))
	for i := 0; i < ld.n; i++ {
		ni[y[i]]++
		for j := 0; j < ld.p; j++ {
			ld.mu.Set(y[i], j, ((ld.mu.At(y[i], j)) + (x.At(i, j))))
		}
	}

	for i := 0; i < ld.k; i++ {
		for j := 0; j < ld.p; j++ {
			ld.mu.Set(i, j, ((ld.mu.At(i, j)) / (float64)(ni[i])))
		}
	}

	// priori is the priori probability of each class
	priori := make([]float64, ld.k)
	for i := 0; i < ld.k; i++ {
		priori[i] = float64(ni[i]) / float64(ld.n)
	}

	// ct is the constant term of discriminant function of each class
	ld.ct = make([]float64, ld.k)
	for i := 0; i < ld.k; i++ {
		ld.ct[i] = math.Log(priori[i])
	}

	// Calculate covariance matrix in 2 steps

	// Step 1: calculate within-class scatter matrix
	// Cw is the within-class scatter matrix initialized as a ld.p x ld.p zero matrix
	Cw := mat.NewSymDense(ld.p, make([]float64, ld.p*ld.p, ld.p*ld.p))

	for i := 0; i < ld.n; i++ {
		for j := 0; j < ld.p; j++ {
			for l := 0; l <= j; l++ {
				Cw.SetSym(j, l, (Cw.At(j, l) + ((x.At(i, j) - ld.mu.At(y[i], j)) * (x.At(i, l) - ld.mu.At(y[i], l)))))
			}
		}
	}
	tol = tol * tol

	// Step 2: calculate between-class scatter matrix
	// Cb is the between-class scatter matrix initialized as a ld.p x ld.p zero matrix
	Cb := mat.NewDense(ld.p, ld.p, make([]float64, ld.p*ld.p, ld.p*ld.p))

	for i := 0; i < ld.k; i++ {
		n := float64(labelMap[i])
		for j := 0; j < ld.p; j++ {
			for l := 0; l < ld.p; l++ {
				Cb.Set(j, l, (Cb.At(j, l) + n*((ld.mu.At(i, j)-colmean[j])*(ld.mu.At(i, l)-colmean[l]))))
			}
		}
	}

	// Solving generalized eigenvalue problem for the matrix
	CwInverse := mat.NewDense(ld.p, ld.p, make([]float64, ld.p*ld.p, ld.p*ld.p))
	CwInverse.Inverse(Cw)
	dotResult := mat.NewDense(ld.p, ld.p, make([]float64, ld.p*ld.p, ld.p*ld.p))
	dotResult.Mul(CwInverse, Cb)
	ld.eigen.Factorize(dotResult, mat.EigenRight)

	// Factorize returns whether the decomposition of the matrix into eigenvectors
	// and eigenvalues succeeded.
	// If the decomposition failed, methods that require a successful factorization will panic
	evals := make([]complex128, ld.p)
	ld.eigen.Values(evals)
	return nil
}

// roRealMatrix returns a dense matrix with just the real parts of the given complex matrix
func toRealMatrix(m mat.CMatrix) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			out.Set(i, j, real(m.At(i, j)))
		}
	}
	return out
}

// getRealVectors returns the right eigen vectors as a real matrix, discarding
// the imaginary parts of the complex vectors
func getRealVectors(e *mat.Eigen) *mat.Dense {
	var complexVectors mat.CDense
	e.VectorsTo(&complexVectors)
	return toRealMatrix(&complexVectors)
}

// Transform performs a transformation on the
// matrix of the input data, which is represented as an ld.n × p matrix x
//
//
// Parameter x is the matrix to be transformed.
// Parameter n is the number of dimensions desired.
// Returns the transformed matrix.
func (ld *LD) Transform(x mat.Matrix, n int) *mat.Dense {
	evecs := getRealVectors(&ld.eigen)
	W := mat.NewDense(ld.p, n, nil)
	for i := 0; i < n; i++ {
		temp := mat.Col(nil, i, evecs)
		W.SetCol(i, temp)
	}
	result := mat.NewDense(ld.n, n, nil)
	result.Mul(x, W)

	return result
}

// Predict performs a prediction based on training data
// to assess which class a certain set of data would be in.
//
// Parameter x is the set of data to classify.
// Returns a prediction for what class the set of data would be in.
//
// Additional details:
// LDA can be used as a supervised learning algorithm to predict
// and classify data based on features extracted from training data.
// LDA reduces dimensionality of the data and performs feature extraction
// to maximize separation between classes.
// Precondition: training data must be labeled and labels must be ints starting
// from 0.
func (ld *LD) Predict(x []float64) (int, error) {

	if len(x) != ld.p {
		return 0, fmt.Errorf("Invalid input vector size")
	}
	var y = 0
	var max = math.Inf(-1)
	d := make([]float64, ld.p)
	ux := make([]float64, ld.p)
	UX := mat.NewDense(len(ux), 1, ux)

	for i := 0; i < ld.k; i++ {
		for j := 0; j < ld.p; j++ {
			d[j] = x[j] - ld.mu.At(i, j)
		}
		evecs := getRealVectors(&ld.eigen)
		Atr := evecs.T()
		D := mat.NewDense(len(d), 1, d)
		UX.Mul(Atr, D) // eigen vector transpose * (measurement - sum of class means)
		var f float64
		evals := make([]complex128, ld.p)
		ld.eigen.Values(evals)
		for j := 0; j < ld.p; j++ {
			f += UX.At(j, 0) * UX.At(j, 0) / cmplx.Abs(evals[j]) // (weighted sum of the result squared) / eigen value
		}
		f = float64(ld.ct[i]) - (0.5 * f)
		if max < f {
			max = f
			y = i
		}
	}
	return y, nil
}

// GetEigen is a getter method for eigen values
//
//
//
// No parameters.
// Returns a mat.Eigen object
func (ld *LD) GetEigen() mat.Eigen {
	return ld.eigen
}
