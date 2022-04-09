# Deep-Learning-model-Coronary-Heart-disease-risk

This notebook aims to use the deep learning methods in order to predict risk of coronary artery disease  from the Framingham study. 
The methodology involves optimising dropout nodes and hidden leayers in deep learning models.

The data from this experiment can be downloaded from https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea

# Table of Contents

[Introduction](#introduction)

- Contextualisation
- Motivation
- Aim

[Methods](#methods)

- Data
- Deep Learning

[Results](#results)

- Extraploratory Data Analysis
- Model Selection
- Chosen Model
- Comparison against simpler models and future experiments

[References](#references)

# Introduction

## Contextualization: Coronary Heart Disease 10-year Prediction

Coronary Heart Disease (CHD) is a chronic and complex disease1. It is classified as complex as an individual’s chance of CHD is controlled by genetic and environmental factors1. In the NHS, GPs predict a patient’s risk of CHD using an ‘Infraction Score’ 2. The patient risk is determined 
from biomarkers known to correlate to CHD risk based on epidemiology studies1. The main biomarkers GPs use are total serum cholesterol, systolic blood pressure, gender smoking habits and family history of heart disease1.The Farmington Heart Study (FHS) is a key study that has improved CHD risk calculation. It began in 1948 and recruited a total of 5,209 men and women whose biomarkers were measured and were followed after 10 years to see which patients suffered CHD3. This prospective study has allowed the development of several risk calculators including QRISK2 and ASSIGN which have helped GPs make better 10-year risk predictions4. 

## Motivation

Coronary Heart Disease (CHD) is estimated to take 17.9 million lives each year, which accounts for around 31% of deaths worldwide. The NHS spends around £7.4 billion yearly treating CHDs. On average, the lifetime risk of CHD for men over 50 years of age is 52% while for women over 50 years it is 39%5. The main issue of CHDs is that it is asymptomatic, individuals often are unaware that they are at high risk of developing CHD1. Estimates of an individual’s long term and short-term risk of CHD will allow preventive measures to be taken to reduce the risk.

## Aim

This project aims to utilise the available FHS data and use deep learning on Tensorflow2.0 to classify patients that had CHD 10 years after biomarkers were measured. This project will aim to determine the ideal number of hidden layer neurons and dropout rates for the model.

# Methods

## Data

The FHS dataset used contains records of 4238 patients and 15 features which include sex, education, smoking habits, glucose measurements, cholesterol measurements and prevalence of certain diseases. The target of the data is binary (whether they had CHD after 10 years).

## Deep Learning 

Deep learning neural networks (DNNs) are capable of solving complex problems with hidden patterns that cannot regularly be in traditional machine learning models. The performance also improves with larger datasets. However, the cost of this is lack of interpretability of the models8. DNNs consist of an input layer, followed by hidden layers and ends on an output layer each consisting of nodes connected to all nodes of the previous layer. All neurons on the first hidden layer are connected to all neurons on the input layer, each containing the value of a feature. The value of the features are multiplied by a weight between -1 and 1. These are summed and become the value of the node of the hidden layer. The weight signifies the relation and importance of a feature. A weight of zero means the feature is essentially ignored. This is repeated until the outcome layer is reached which assigns the prediction to a class. For binary classification, the outcome layer consists of one node which is activated by a sigmoid function. During training, a number of epochs are set that defines the number times that model will work through the entire training set. For each epoch, the loss is measured as a score and an optimizer updates the weights and biases to reduce the loss. For binary classification, Binary cross entropy is the most often used function.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/deep_learning_formula.png)

Binary cross entropy computes loss as the vertical distance between the class (0 or 1) from the decision of the sigmoid function classifier10. 

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/cross_entropy.png)

The negative log loss is then calculated, this is to better penalise bad predictions, as the loss increases exponentially with poor decisions.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/logloss.png)

After the loss is calculated the optimiser (Adam), which is a form of stochastic gradient descent, will change the weights of the model to reduce the loss. This is done by gradient descent. Gradient descent takes the change in loss of the current and previous model to determine the direction to improve the model. This creates a quadratic function where the minimum point (gradient = 0) is the most optimal model with the least loss. Adam measures the gradient of the error and change in bias to decrease it to zero. The rate in how far a step it takes for each iteration is determined the learning rate or alpha. If alpha is set too small gradient descent will take too long, if it is too big it will overshoot the minimum point.10

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/gradient_descent.png)

The number of hidden layers and the neurons in the layers determine the complexity of the model. Often DNNs suffer overfitting, where the model fails to generalize to new data, which is often characterised by a high training accuracy but a poor test accuracy. Adding dropouts to the hidden layers is a simple method to overcome overfitting, the dropout rate determines the fraction of neurons that will randomly be deleted in the hidden layer11.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/dropouts.png)

For this experiment, three hidden layers will be used which connect to a final output layer with one neuron where the sigmoid function is implemented to address the prediction to the binary class. The hidden layers will be tested with neurons of 30,35,40,45,50,55 and 60 in hidden layers of 1 to 4. This will be redone with a drop out rate of 0.5 on each hidden layer and double the neurons.

# Results

## Extraploratory Data Analysis

The database provided contained a total of 4238 and 16 features which included binomial data (diabetes, current smoker, previous stroke), ordinal data (education, cigarettes per day) and quantitative data (blood pressure, total cholesterol). The box plot below shows the spread of the quantitative data.

The education feature was dropped as there were too many missing values. Then, rows with missing values were dropped as well. The data was then normalised in order to place the features on the same scale of 0 to 1. The data was split into a training, validation, and test set. The training set was 75% of the data and the test set was 25%. The validation set was 10% of the training data. As the data is imbalanced, the training and validation sets were oversampled to match the majority class in order to prevent bias.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/EDA.png)

## Model Selection

The validation set will be used to determine the model with the ideal number of neurons for each hidden layer number. This will be tested then with the test set. Receiving Operator Curves (ROC) will be plotted alongside confusion matrixes and sensitivity, specificity, positive predictive value (PPV) and negative predictive value (NPV) to assess each model.

## Chosen Model

The figures below show the training and validation set accuracies for each hidden layers for each neuron number. Based on the graphs the models chosen for further testing are on the table below.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/Model_selection1.png)

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/Model_selection2.png)

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/Confusion_matrix.png)

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/chosen_models.png)

## Comparison against simpler models and future experiments

A simple Random Forest model was created with default scikit-learn parameters to test against simpler models. The results are below.

![](https://github.com/YLumad/Deep-Learning-model-Coronary-Heart-disease-risk/blob/main/images/RandomForest.png)

Despite a ROC curve above the 50% threshold, it is clear that the model is bias towards the negative class. Although the model is untuned, the root of the issue of this problem is regarding the lack of positive cases. The deep learning neural networks outperformed the random forest model, even after oversampling the random forest model was extremely biased towards the negative class. The model seems to perform in a different manner as the Orfanoudaki et al. paper. Where the models had low negative predictive power but high positive predictive power meaning it better classified the minority class7.

The Buenza et al. 2019 study had similar results, a high negative predictive power and low positive predictive power. The researchers also used oversampling to balance the classes and concluded that the main issue was the nature of clinical data, it is difficult to guarantee the results as the data can be unreliable6. It is also prone to missing values as we have seen in this study. Buenza et al. 2019 opted to change the missing values to the mean, this was not done in this study as the reason behind the value being missing may have been from patient decision and not from randomness (e.g. a person who is obese is less willing to disclose their weight). This is common in health databases and changing a missing value that is likely to have high variance to the mean may not be worth it.

Overall this study shows that adding dropouts to hidden layers is an effective method to handle overtraining. All the models did better at predicting the positive class except at 4 hidden layers. A follow up experiment would be to undersample the data to match the class counts in the training set as opposed to oversampling. Oversampling the data may have caused the noise in the imbalanced class to be repeated multiple times and therefore had weight on decision making. Also, it appears that the model can still benefit with further neurons and hidden layers as the accuracy in Figure 8 suggest that the validation set accuracy was still improving. However this experiment itself was already computationally expensive. The main issue with deep learning is that it is difficult to assess the weight of the features. The cost of a having a machine learning model that can detect complex pattern between features is not knowing how the features interact. The figure below shows the optimal decision tree of a similar study by Glienke (2020)13.

Figure 12. Optimised decision tree classifier. 
13
Glinke also had the same issue of low PPV as the models identified less than 15 true positives. What was interesting was that the logistic regression model found being a male as the primary feature for decision making, which was not seen in the decision tree model.
Figure 13. Optimised linear regression model.13
Lastly, it appears that ROC AUC measurement is not ideal for the imbalanced dataset. The random forest showed some predictive power despite being highly biased and only 0.14 sensitivity. Other studies also used ROC AUC as the main measurement for accuracy, the Precision-Recall Curves may have been a more suitable decision as it puts more weight on the minority class14

# Conclusion

The FHS is a rich dataset of 16 features of patients. Its main strength is being a prospective 10 year follow up for CHD. However, the drawback of this is that there is a major class imbalance of 644 CHD patients and 3594 controls. This study aimed to optimising hidden layer and neuron numbers with and without dropouts to classify whether people will suffer from CHDs based on biomarkers measured 10 years prior. The models without dropouts had sensitivity rates of 0.47-0.54 and specificity rates of 0.70-0.74, while the dropout models had rates of 0.65-0.59 and 0.60-0.70. The NPV and PPV rates were similar across the models (around 0.88 and 0.25). The models with dropouts seemed to perform better in terms of identifying the minority class but with a substantial loss of identifying the majority class. The main issue appears to be the lack of people with CHD, causing bias in decision making, this study attempted to alleviate that by oversampling the training set. The models still poorly classed the CHD group, this was seen to be similar in other studies that oversampled the training set. This suggests the general difficulty in identifying patient risk of CHDs 10 years prior.


