package lda

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"image/color"
	"io"
	"log"
	"os"
	"strconv"
	"testing"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func TestLinearDiscriminant(t *testing.T) {
	// Threshold for detecting zero variances
	var ld LD
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

	ok := ld.LinearDiscriminant(dataMatrix, labelsNumbers) // Calling LinearDiscriminant on Iris data
	if ok == nil {
		fmt.Println("Call to LDA successful")
		numDims := 2
		result := ld.Transform(dataMatrix, numDims)
		// Graphing results of the transformation
		PlotLDA(result, labelsNumbers, "Iris-data-LDA-graph.png", "LDA: Iris Dataset")
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

// PlotLDA plots the LDA transformation on an (X,Y) plane and returns a PNG
// of the graph, which is saved in the same directory as the source code
func PlotLDA(Data *mat.Dense, labels []int, imageTitle string, graphTitle string) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = graphTitle
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	scatterData := matrixToPoints(Data)
	sc, err := plotter.NewScatter(scatterData)
	if err != nil {
		log.Panic(err)
	}

	sc.GlyphStyleFunc = func(i int) draw.GlyphStyle {
		r := (map[bool]uint8{true: 128, false: 0})[labels[i]&(1<<2) != 0]
		g := (map[bool]uint8{true: 128, false: 0})[labels[i]&(1<<1) != 0]
		b := (map[bool]uint8{true: 128, false: 0})[labels[i]&1 != 0]
		a := uint8(255)
		color := color.RGBA{r, g, b, a}
		markers := [7]draw.GlyphDrawer{
			draw.CrossGlyph{},
			draw.CircleGlyph{},
			draw.PyramidGlyph{},
			draw.TriangleGlyph{},
			draw.SquareGlyph{},
			draw.RingGlyph{},
			draw.PlusGlyph{},
		}
		return draw.GlyphStyle{Color: color, Radius: vg.Points(3), Shape: markers[labels[i]%7]}
	}
	p.Add(sc)
	p.Add(plotter.NewGrid())

	if err := p.Save(8*vg.Inch, 5*vg.Inch, imageTitle); err != nil {
		panic(err)
	}
}

func matrixToPoints(data *mat.Dense) plotter.XYer {
	r, c := data.Dims()
	if c != 2 {
		panic("Matrix must have 2 columns (2D matrix only)")
	}
	pts := make(plotter.XYs, r)
	for i := 0; i < r; i++ {
		pts[i].X = data.At(i, 0)
		pts[i].Y = data.At(i, 1)
	}
	return pts
}
