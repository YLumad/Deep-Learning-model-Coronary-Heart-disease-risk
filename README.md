# Deep-Learning-model-Coronary-Heart-disease-risk
This notebook aims to use the deep learning methods in order to predict risk of coronary artery disease  from the Framingham study. 
The methodology involves optimising dropout nodes and hidden leayers in deep learning models.

The data from this experiment can be downloaded from https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea

# Table of Contents

[Introduction](#introduction)

[Methods](#methods)

[Results](#results)

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


