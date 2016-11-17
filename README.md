# spark-ml-predict-uk-house-prices

This is an example to show how Spark ML could be used to predict house prices in the UK. 

Decision Trees are used for their simplicity and readbility. 

The data used to train the model are extracted from [Data UK](https://data.gov.uk/dataset/land-registry-monthly-price-paid-data) for the year 2015. 

Three features are used to predict the house prices. Since it's a regression problem, all features are converted to continous values to ease the regression process.

## The Model

The model was trained for the whole data set (~3.7GB) with an average Mean Square Error of `3.1733904673297367E10`.

Here is a readable view of the model.

Feature 0: Indicates whether the house is located.
```
1 = London , 0 = Other
```
Feature 1: Indicates which property type it is. 
```
0 = Detached, 1 = Semi-Detached, 2 = Terraced, 3 = Flats/Maisonettes, 4 = Other
```
Feature 2: Indicates the duration of tenture. 
```
Relates to the tenure: 0 = Freehold, 1= Leasehold, 2 = Unknown?.
```
It's worth noting that `U` value exists in the data set but wasn't documented by the data provider.

Learned regression tree model:
```
DecisionTreeModel regressor of depth 5 with 35 nodes
  If (feature 1 <= 3.0)
   If (feature 0 <= 0.0)
    If (feature 1 <= 0.0)
     If (feature 2 <= 0.0)
      Predict: 243699.06697266884
     Else (feature 2 > 0.0)
      If (feature 2 <= 1.0)
       Predict: 177431.79831382376
      Else (feature 2 > 1.0)
       Predict: 202724.02564102566
    Else (feature 1 > 0.0)
     If (feature 1 <= 1.0)
      If (feature 2 <= 0.0)
       Predict: 142017.86041304827
      Else (feature 2 > 0.0)
       Predict: 106310.48170044324
     Else (feature 1 > 1.0)
      If (feature 1 <= 2.0)
       Predict: 114960.67808362727
      Else (feature 1 > 2.0)
       Predict: 135064.69324986602
   Else (feature 0 > 0.0)
    If (feature 1 <= 0.0)
     If (feature 2 <= 0.0)
      Predict: 872222.3417896043
     Else (feature 2 > 0.0)
      If (feature 2 <= 1.0)
       Predict: 671787.7266666667
      Else (feature 2 > 1.0)
       Predict: 338333.3333333333
    Else (feature 1 > 0.0)
     If (feature 1 <= 2.0)
      If (feature 1 <= 1.0)
       Predict: 435987.88375123806
      Else (feature 1 > 1.0)
       Predict: 377497.5393950943
     Else (feature 1 > 2.0)
      If (feature 2 <= 0.0)
       Predict: 319726.48873538786
      Else (feature 2 > 0.0)
       Predict: 278006.80455810396
  Else (feature 1 > 3.0)
   If (feature 0 <= 0.0)
    If (feature 2 <= 0.0)
     Predict: 1376057.377781664
    Else (feature 2 > 0.0)
     Predict: 1455298.5163765675
   Else (feature 0 > 0.0)
    If (feature 2 <= 0.0)
     Predict: 5168805.568297655
    Else (feature 2 > 0.0)
     Predict: 2764222.786941581
```

## How to Run 

Make the project and run test cases.

```
mvn clean install
```
Run the program
```
mvn exec:java
```

## License

This repo is released under the Apache 2.0 license.
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
