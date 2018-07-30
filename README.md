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

*See `lda_test.go` for example usage* <br/>
The library contains a predefined struct `LD` that can be used to access LDA methods. <br/>
Important note about input data: LDA is a supervised learning technique so all training data needs to be labeled. <br/>
<br/>
The first step would be to make a variable of type `LD` and call `LinearDiscriminant` by passing in a matrix of input data and an array of labels that correspond to the data. If that call is successful, then you can use other LDA methods, such as `Tranform` and `Predict`. <br/>
Example: <br/>
```
var ld LDA.LD
err := ld.LinearDiscriminant(dataMatrix, labels)
if err == nil {
  fmt.Println("Call to LDA successful")
  numDimensions := 2 // number of dimensions to reduce to
  result := ld.Transform(dataMatrix, numDimensions)
  // use result
} else {
	fmt.Printf("Call to LDA failed: %v\n", err)
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
