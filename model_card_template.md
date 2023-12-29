# Model Card

## Model Details
This binary classification model was created using a random forest classifier. No hyperparameter tuning was performed. 

## Intended Use
The intended use of this model is to take various demographic and employment inputs and predict whether the individual's
income above or below/equal to $50,000.

## Training Data
The training data contains income census data gathered by UC Irvine Machine Learning Repository 
(https://archive.ics.uci.edu/dataset/20/census+income) in a csv file. The data within contained 32,561 rows and 15
columns. Feature information is available at link provided. The 8 categorical preprocessed with OneHotEncoder 
and LabelBinarizer. The census data was split to 85:15, 85% of the data was used for training and 15% used for testing.
Random state was set to 42 for reproducibility.

## Evaluation Data
15% of the census data was used for model testing. The testing data was preprocessed the same as the training data.

## Metrics
The model was evaluated on Precision, Recall  and F1-score
Precision: 0.7550 | Recall: 0.6539 | F1: 0.7008

## Ethical Considerations
The census data itself may have biases leading to biases in the predictions. Consideration should be given when using
model as it may not be an accurate representation of income levels.

## Caveats and Recommendations
Model was trained on census data from 1994. Consideration should be used as training data is old and may not reflect
many of the socioeconomic changes that have occurred since 1994.

Biases of the census data should be taken into account, whether from the time period or from gathering techniques.