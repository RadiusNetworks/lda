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
The first step would be to call `LinearDiscriminant`
