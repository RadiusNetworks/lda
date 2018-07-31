# Linear Discriminant Analysis in Go

Linear Discriminant Analysis is a powerful and commonly used dimensionality reduction and classification technique used in statistics 
and machine learning. It extracts components from an input dataset in a way that maximizes class separability 
while minimizing the variance within each class. A key assumption here is that the dataset needs to be
normally distributed. In the world of machine learning, LDA can be used as a classifier algorithm, which this 
library also provides.

## Getting Started 

Import `"github.com/RadiusNetworks/lda"`

### Dependencies

`math`
`sort`
`gonum.org/v1/gonum/mat`

## Usage

The library contains a predefined struct `LD` that can be used to access LDA methods. <br/>
Important note about input data: LDA is a supervised learning technique so all training data needs to be labeled. <br/>
<br/>
The first step would be to make a variable of type `LD` and call `LinearDiscriminant` by passing in a matrix of input data and an array of labels that correspond to the data. If that call is successful, then you can use other LDA methods, such as `Tranform` and `Predict`. <br/>
<br/>
Example: Iris dataset <br/>
```
// Load Iris data from iris/iris.data
// Process data (See lda_test.go for code example)

// Create a matrix and fill it with iris data
dataMatrix := mat.NewDense(numberOfRows, numberofColumns, yourDataset)

// Create an array ([]int) of labels for your dataset
var labels []int
for yourLabel := range labelsFromYourDataset {
	labels = append(labels, yourLabel)
}

// Instantiate an LD object and call LinearDiscriminant to check if input data follows preconditions
var ld LDA.LD
err := ld.LinearDiscriminant(dataMatrix, labels)

if err == nil {
  // If the call is successful, you can now use other methods
  numDimensions := 2 // number of dimensions to reduce to
  result := ld.Transform(dataMatrix, numDimensions)
  
  // We can graph the result of the transformation on an XY plane
  PlotLDA(result, labels, "LDA Plot.png")
  
  // We can use the result of the transformation to classify test data
  // *See section on method Predict below*
  
} else {
  // handle error
}

...

// An example graphing function that takes in the result of the transformation, the array of labels and a title for the image it will create
func PlotLDA(Data *mat.Dense, labels []int, imageTitle string) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "LDA Transform"
	p.X.Label.Text = "X-axis"
	p.Y.Label.Text = "Y-axis"

	scatterData := matrixToPoints(Data)
	sc, err := plotter.NewScatter(scatterData)
	if err != nil {
		log.Panic(err)
	}
	
	// Iterating over various markers and colors to make the graph more visually appealing
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
	
	// Saving the resulting graph as a PNG file
	if err := p.Save(12*vg.Inch, 8*vg.Inch, imageTitle); err != nil {
		panic(err)
	}
}
```

### Using `Predict` to classify test data

Method `Predict` takes in a set of data and returns a number (Int), a prediction for what class the set of data would be in. Below is an example of a test that checks if Predict is classifying data correctly. <br/>
Example: Iris test data <br/>
(See lda_test.go for the complete implementation of this example) <br/>
```
// Create test cases with test data and corresponding labels (classes that you expect the data points to be in)
// Call LinearDiscriminant with the test data and the labels as arguments
// Call Transform
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
```

## Tests

We provide a sample test file that tests both the dimensionality reduction and the classification features of the algorithm. The test uses the famous Iris dataset, which can be found here: https://archive.ics.uci.edu/ml/datasets/Iris

## Credits and Acknowledgements

The implementation of the LDA algorithm is based on a Java version provided by https://github.com/haifengl/smile <br/>
Additional resources used: https://sebastianraschka.com/Articles/2014_python_lda.html <br/>
Created with the help of Tim Judkins (@b0tn3rd) and Eleanor Nuechterlein (@Eleanor2)

## Contributing and License
LDA-go is Apache-2.0 licensed. Contributions are welcome.
