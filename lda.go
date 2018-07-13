package lda

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// LD is a type for computing and extracting the linear discriminant analysis of a
// matrix. The results of the linear discriminant analysis are only valid
// if the call to LinearDiscriminant was successful.
type LD struct {
	n, p  int //n = row, p = col
	k     int
	ct    []float64  //Constant term of discriminant function of each class
	mu    *mat.Dense //Mean vectors of each class
	svd   *mat.SVD
	ok    bool
	eigen mat.EigenSym //Eigen values of common variance matrix
}

// LinearDiscriminant performs a linear discriminant analysis on the
// matrix of the input data which is represented as an n×p matrix x where each
// row is an observation and each column is a variable.
//
//
// Parameter x is the training samples
// Parameter y is the training labels in [0,k)
// where k is the number of classes
// Returns whether the analysis was successful
func (ld *LD) LinearDiscriminant(x mat.Matrix, y []int) (ok bool) {
	ld.n, ld.p = x.Dims()
	if y != nil && len(y) != ld.n {
		panic("The sizes of X and Y don't match")
	}
	var labels []int
	var labelMap = map[int]bool{}
	for _, label := range y {
		if !labelMap[label] {
			labelMap[label] = true
			labels = append(labels, label)
		}
	}
	// Create a new array with labels and go through the array of y values and if
	// it doesnt exist then add it to the new array
	sort.Ints(labels)

	if labels[0] != 0 {
		panic("Label does not start from zero")
	}
	for i := 0; i < len(labels); i++ {
		if labels[i] < 0 {
			panic("Negative class label")
		}
		if i > 0 && labels[i]-labels[i-1] > 1 {
			panic("Missing class")
		}
	}
	// Tol is a tolerence to decide if a covariance matrix is singular (det is zero)
	// Tol will reject variables whose variance is less than tol
	var tol = 1E-4
	// k is the number of classes
	ld.k = len(labels)
	if ld.k < 2 {
		panic("Only one class.")
	}
	if tol < 0.0 {
		panic("Invalid tol")
	}
	if ld.n <= ld.k {
		panic("Sample size is too small")
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

	// C is a matrix of zeros with dimensions: ld.p x ld.p
	C := mat.NewSymDense(ld.p, make([]float64, ld.p*ld.p, ld.p*ld.p))

	// Class mean vectors
	// mu is a matrix with dimensions: k x ld.p
	ld.mu = mat.NewDense(ld.k, ld.p, make([]float64, ld.k*ld.p, ld.k*ld.p))
	for i := 0; i < ld.n; i++ {
		ni[y[i]] = ni[y[i]] + 1
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
		priori[i] = (float64)(ni[i] / ld.n)
	}

	// ct is the constant term of discriminant function of each class
	ld.ct = make([]float64, ld.k)
	for i := 0; i < ld.k; i++ {
		ld.ct[i] = math.Log(priori[i])
	}

	for i := 0; i < ld.n; i++ {
		for j := 0; j < ld.p; j++ {
			for l := 0; l <= j; l++ {
				C.SetSym(j, l, (C.At(j, l) + ((x.At(i, j) - colmean[j]) * (x.At(i, l) - colmean[l]))))
			}
		}
	}

	tol = tol * tol

	for j := 0; j < ld.p; j++ {
		for l := 0; l <= j; l++ {
			C.SetSym(j, l, ((C.At(j, l)) / (float64)(ld.n-ld.k)))
			C.SetSym(l, j, C.At(j, l))
		}
		if C.At(j, j) < tol {
			panic("Covarience matrix (variable %d) is close to singular")
		}
	}

	// Factorize returns whether the decomposition succeeded
	// If the decomposition failed, methods that require a successful factorization will panic
	ld.eigen.Factorize(C, true)
	return true
}

// Transform performs a transformation on the
// matrix of the input data which is represented as an ld.n × p matrix x
//
//
// Parameter x is the matrix being transformed
// Returns the result (transformed) matrix
func (ld *LD) Transform(x mat.Matrix) *mat.Dense {
	values := make([]float64, ld.p*ld.p, ld.p*ld.p)
	evecs := mat.NewDense(ld.p, ld.p, values)
	evecs.EigenvectorsSym(&ld.eigen)
	result := mat.NewDense(ld.n, ld.p, make([]float64, ld.n*ld.p, ld.n*ld.p))
	result.Mul(x, evecs)
	return result
}

// Predict performs a prediction to assess which zone a certain
// set of data would be in
//
//
// Parameter x is the set of data
// Returns which zone the set of data would be in
func (ld *LD) Predict(x []float64) int {
	if len(x) != ld.p {
		panic("Invalid input vector size")
	}
	var y int
	var max = math.Inf(-1)
	d := make([]float64, ld.p)
	ux := make([]float64, ld.p)
	for i := 0; i < ld.k; i++ {
		for j := 0; j < ld.p; j++ {
			d[j] = x[j] - ld.mu.At(i, j)
		}
		var f float64
		evals := make([]float64, ld.p)
		ld.eigen.Values(evals)
		for j := 0; j < ld.p; j++ {
			f += ux[j] * ux[j] / math.Abs(evals[j])
		}
		f = ld.ct[i] - 0.5*f
		if max < f {
			max = f
			y = i
		}
	}
	return y

}

// GetEigen is a getter for eigen values
//
//
//
// No parameters
// Returns eigen values
func (ld *LD) GetEigen() mat.EigenSym {
	return ld.eigen
}
