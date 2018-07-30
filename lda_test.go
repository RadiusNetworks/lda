package lda

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLinearDiscriminant(t *testing.T) {
	// Threshold for detecting zero variances
	const epsilon = 1e-15

	// Iris dataset training file
	trainFile, _ := os.Open("iris/iris.data")
	rTrain := csv.NewReader(bufio.NewReader(trainFile))
	rTrain.Comma = ','
	var trainingDataText []string
	var trainingDataNumbers []float64
	var labels []string
	var labelsNumbers []int
	var numRows int
	for {
		trainRecord, err := rTrain.Read()
		if len(trainRecord) != 0 {
			numRows++
			trainingDataText = append(trainingDataText, trainRecord[0:4]...)
			labels = append(labels, trainRecord[4])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
	}
	for _, arg := range trainingDataText {
		if n, err := strconv.ParseFloat(arg, 64); err == nil {
			trainingDataNumbers = append(trainingDataNumbers, n)
		}
	}
	dataMatrix := mat.NewDense(numRows, 4, trainingDataNumbers)

	// Map of labels to ints
	var m = map[string]int{}
	i := 0
	for _, value := range labels {
		if _, ok := m[value]; !ok {
			m[value] = i
			i++
		}
	}

	for value := range labels {
		labelsNumbers = append(labelsNumbers, m[labels[value]])
	}

tests:
	for i, test := range []struct {
		data        mat.Matrix
		labels      []int
		testPredict *mat.Dense
		wantClass   []int
		wantVecs    *mat.Dense
		wantVars    []float64
		epsilon     float64
	}{
		{
			data:   dataMatrix,
			labels: labelsNumbers,
			testPredict: mat.NewDense(3, 4, []float64{
				5.0, 3.3, 1.4, 0.2, // Setosa
				5.1, 2.5, 3.0, 1.1, // Versicolor
				7.7, 3.0, 6.1, 2.3, // Virginica
			}),
			wantVecs: mat.NewDense(4, 4, []float64{
				0.2049, -0.008982, 0.8846, -0.03005,
				0.3871, -0.589, -0.2924, 0.3471,
				-0.5465, 0.2543, -0.2664, 0.4156,
				-0.7138, -0.767, -0.247, -0.8402,
			}),
			wantVars:  []float64{32.27195779972984 + 0i, 0.2775668638400518 + 0i, -5.788566151963261e-16 + 0i, -1.7908428048920807e-14 + 0i},
			wantClass: []int{2, 0, 1},
			epsilon:   1e-12,
		},
	} {
		var ld LD
		for j := 0; j < 1; j++ {
			err := ld.LinearDiscriminant(test.data, test.labels)
			if err != nil {
				t.Log(err)
				t.Errorf("unexpected SVD failure for test %d use %d", i, j)
				continue tests
			}
			numDims := 2
			result := ld.Transform(test.data, numDims)
			r, _ := test.testPredict.Dims()
			for k := 0; k < r; k++ {
				c, _ := ld.Predict(test.testPredict.RawRowView(k))
				if c != test.wantClass[k] {
					t.Errorf("unexpected prediction result %v got:%v, want:%v", k, c, test.wantClass[k])
				}
			}
			values := make([]string, ld.p)
			for j := 0; j < ld.n; j++ {
				row := result.RawRowView(j)
				for k := 0; k < numDims; k++ {
					values[k] = fmt.Sprintf("%.4f", row[k])
				}
			}
		}
	}
}

func checkError(message string, err error) {
	if err != nil {
		log.Fatal(message, err)
	}
}
