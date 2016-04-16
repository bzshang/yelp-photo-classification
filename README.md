# yelp-photo-classification

This document outlines my 4th place solution for the [Kaggle Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification/) competition.

## Outline

We fine-tune the pre-trained Inception V3 network provided by mxnet [here](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md).

We modify the symbol file [here](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol_inception-v3.py) by renaming the fully-connected layer. This prevents mxnet from trying to initialize this layer with pre-trained weights. We also reduce the number of output classes from 1000 to 9 (as there are 9 business labels to predict) and change the output to LogisticRegressionOutput (independent sigmoids) which is appropriate for multi-label learning, as the output probabilties are not constrained to add to 1.

Using this modified architecture, we initialize the model with pre-trained weights, except for the last layer which was renamed and modified.

We train the network for 5 epochs through the training set, using random crop, mirror, and scaling of the images. An AWS Spot instance (g2.2xlarge) was used with cuDNN.

The features (1024 total) from the global_pool layer (next to last) are then extracted.

Business features were chosen to be the average of their image features.

These features are used as input into classical machine learning (ML) models.
These models include support vector classification, logistic regression, and random forest. We use one-vs-rest methodology for the multi-label problem.

The class label probabilities from the 3 ML models are averaged and we use a threshold of 0.46 (determined from cross-validation) for selecting a label.

Finally, a majority vote classifier is used taking features extracted from epochs 3 through 5.

## Factors improving score

Fine-tuning of the network resulted in an improvement of 0.04 in local CV tests. On the public leaderboard, this resulted in a score around 0.81.

Interestingly, adding a random forest classifier resulted in about a 0.01 improvement. Ensembling added an additional 0.01 improvement.

On the last day, I found that decreasing the regularization parameter in the SVC model helped increase the score. One of the submissions did okay on the public leaderboard but would have been first on the private leaderboard.
Most likely, though, it would have been a lucky submission if chosen, but ensembling more of these types of models would have resulted in more robust, reliable improvements.

## Scripts

This part will outline the scripts I used.

### 1. Generate the image label and record file

I used the ImageRecordIter from mxnet, which has quite good performance. The record file generator first requires a .lst file which is what create_img_list.py does.
After creating the .lst file, generate the record file following the steps [here](http://myungjun-youn-demo.readthedocs.org/en/latest/python/io.html). 
Use resize=299 label_width=9.

### 2.  Train the Inception V3 network

The script train_inception.py fine-tunes the Inception V3 model pre-trained by mxnet.

### 3.  Extract the image features

The script get_image_features.py extracts the image features from the global_pool layer (next to last) of the fine-tuned model.

### 4.  Get the business features

The business features are the average of the image features. The scripts get_biz_features*.py extract these features.

### 5. Train the machine learning models

Using the business features and labels, train three different ML models (support vector classification, logistic regression, and random forest). Use the one-vs-rest approach for multi-label classification. I used scikit-learn for these models in train_ml.py.

### 6. Generate predictions

Using the models, generate predictions in test_ml.py. The predictions use the average probabilities from the 3 ML models and a threshold of 0.46 for assigning labels. The threshold was chosen from local CV runs.

### 7. Generate ensemble majority vote classifier

After generating predictions from features extracted from different epochs, the script merge_submissions.py generates an majority vote classifier. I found ensembled model scores were generally more reliable, in terms of private/public scores being closer.
