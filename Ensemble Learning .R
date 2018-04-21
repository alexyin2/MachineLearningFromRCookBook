# Introduction of Ensemble Learning
##Ensemble learning is a method for combining results produced by different learners into one format, with the aim of producing better classification results and regression results.
##A single classifier may be imperfect, as it misclassify data in certain categories.
##As not all classifiers are imperfect, a better approach is to average the results by voting.
##In other words, if we average the prediction results of every classifier with the same input, we may create a superior model compared to using an individual method.
##---
##In esemble learning, bagging, boosting, and randomforest are the three most common methods.
##1. Bagging is a voting method, which first uses Bootstrap to generate a different training set, and then uses the training set to make different base learners. 
##The bagging method employs a combination of base learners to make a better prediction.
##2. Boosting is similar to the bagging method. However, what makes boosting different is that it first constructs the base learning in sequence, where each successive learner is built for the prediction residuals of the preceding learner. 
##With the means to create a complementary learner, it uses the mistakes made by previous learners to train the next base learner.
##3. Random Forest uses the classification results voted from many classification trees. 
##The idea is simple; a single classification tree will obtain a single classification result with a single input vector. 
##However, a random forest grows many classification trees, obtaining multiple results from a single input. 
##Therefore, a random forest will use the majority of votes from all the decision trees to classify data or use an average output for regression.
##4. Stacking or Super Learner algorithm multiple models are used and stacked over one another, predictions are passed from one layer to another and to the top layer. 
##It is a loss-based supervised learning method.


# Using the Super Learner Algorithm
##Super Learner is an algorithm that uses various learning prediction models and finds the optimal combination of collection of algorithms. It learns best optimal fits. 
##Ensemble methods use multiple learning algorithms to obtain better performance.
library(SuperLearner)
##To list all the available wrappers for predictions and screening, execute the following:
listWrappers()
data(Boston, package = "MASS")
training = sample(nrow(Boston), 100)
X = Boston[training, ]
X_Hold = Boston[-training, ]
Y = Boston$medv[training]
Y_Hold = Boston$medv[-training]
##We can fit single model using the SuperLearner
sl_glm = SuperLearner(Y = Y, X = X, family = gaussian(), SL.library = "SL.glmnet")
##Y means the outcome in the training data set, must be a numeric vector
##X means the predictor variables in the training data set, usually a data.frame
sl_glm
sl_randomf = SuperLearner(Y = Y, X = X, family = gaussian(), SL.library = "SL.randomForest")
sl_randomf
##Risk shows model accuracy or performance--the lower the risk, the better the model.
##The coefficient column tells the weight of a model.


# Using Ensemble to Train and Test
##We can fit more than one model in Super Learner and it will tell which is best from all the applied models. It also creates a weighted average for all the models.
data(Boston, package = "MASS")
training = sample(nrow(Boston), 100)
X = Boston[training, ]
X_Hold = Boston[-training, ]
Y = Boston$medv[training]
Y_Hold = Boston$medv[-training]
library(SuperLearner)
sl_models = c("SL.xgboost", "SL.randomForest", "SL.glmnet", "SL.nnet", "SL.rpartPrune", "SL.lm", "SL.mean")
superlearner = SuperLearner(Y = Y, X = X, family = gaussian(), SL.library = sl_models)
superlearner
r = CV.SuperLearner(Y = Y, X = X, family = gaussian(), SL.library = sl_models)
plot(r)
##We can predict using the data we removed from the training set to see if its predicted correctly or not (recall the X and X_Hold variable we created in beginning):
pred = predict(superlearner, X_Hold, onlySL = T)
pred
sum(abs(Y_Hold - pred$pred) < 10 ) / length(Y_Hold)
##We can see that we compare the predicted values and the real value. We set a range of 20 and find out that the accuracy is about 0.84


# Classifying Data with the Bagging Method using the adabag Package, which uses classification trees as single classifiers
##The adabag package implements both boosting and bagging methods.
##We first prepare the data we need to use.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
##Next, we can use the bagging function to train a training dataset
set.seed(2)
churn.bagging = bagging(churn ~ ., data = trainset, mfinal = 10)
##Access the variable importance from the bagging result
churn.bagging$importance
##After generating the classification model, we can use the predicted results from the testing dataset:
churn.predbagging = predict.bagging(churn.bagging, testset)
##From the predicted results, we can obtain a classification table:
churn.predbagging$confusion
##The definition of bagging is as follows: given a training dataset of size n, bagging performs Bootstrap sampling and generates m new training sets, Di, each of size n. 
##Finally, we can fit m Bootstrap samples to m models and combine the result by averaging the output (for regression) or voting (for classification)


# Performing cross-validation with the bagging method
##To assess the prediction power of a classifier, you can run a cross-validation method to test the robustness of the classification model.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
churn.baggingcv = bagging.cv(churn ~ ., data = trainset, v = 10, mfinal = 10)
##We can then obtain the confusion matrix from the cross-validation results:
churn.baggingcv$confusion
##We can retrieve the minimum estimation errors from the cross-validation results:
churn.baggingcv$error


# Classifying Data with the Boosting Method with adabag Package, which use classification trees as single classifiers.
##Similar to the bagging method, boosting starts with a simple or weak classifier and gradually improves it by reweighting the misclassified samples.
##Thus, the new classifier can learn from previous classifiers.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
set.seed(2)
churn.boost = boosting(churn ~ ., data = trainset, mfinal = 10, coeflearn = "Freund", boos = F, control = rpart.control(maxdepth = 3))
##We can then make a prediction based on the boosted model and testing dataset:
churn.boost.pred = predict.boosting(churn.boost, testset)
churn.boost.pred$confusion
churn.boost.pred$error
##The idea of boosting is to boost weak learners (for example, a single decision tree) into strong learners.
##Assuming that we have n points in our training dataset, we can assign a weight, Wi (0 <= i <n), for each point. Then, during the iterative learning process (we assume the number of iterations is m), we can reweigh each point in accordance with the classification result in each iteration.
##If the point is correctly classified, we should decrease the weight. Otherwise, we increase the weight of the point.


# Performing Cross-validaton with the boosting method
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
##First, we can use boosting.cv to cross-validate the training dataset:
churn.boostcv = boosting.cv(churn ~ ., v = 10, data = trainset, mfinal = 5, control = rpart.control(cp = 0.01))
##we can then obtain the confusion matrix from the boosting results:
churn.boostcv$confusion
##Finally, we can retrieve the average errors of the boosting method:
churn.boostcv$error


# Classifying Data with Gradient Boost using gbm package
##Gradient boosting ensembles weak learners and creates a new base learner that maximally correlates with the negative gradient of the loss function.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(gbm)
##The gbm function only uses responses ranging from 0 to 1, so we have to transform yes / no responses into 1 / 0
trainset$churn = ifelse(trainset$churn == "yes", 1, 0)
##Next, we can use the gbm function to train a training dataset:
set.seed(2)
churn.gbm = gbm(churn ~ ., data = trainset, distribution = "bernoulli", n.trees = 1000, interaction.depth = 7, shrinkage = 0.01, cv.folds = 3)
##Explaining parameters in churn.gbm:
##shrinkage means the learning rate of the step size reduction.
##cv.folds means the number of cross-validations.
##---
##Then, we can obtain the summary information from the fitted model:
summary(churn.gbm)
##We can obtain the best iteration using cross-validation:
churn.iter = gbm.perf(churn.gbm, method="cv")
##Explaining the plot: 
##The function further generates two plots, where the black line plots the training error and the green one plots the validation error. 
##The error measurement here is a bernoulli distribution, which we defined earlier in the training stage.
##The blue dash line on the plot shows where the optimum iteration is.
##---
##Then, we can retrieve the odd value of the log returned from the Bernoulli loss function
churn.predict = predict(churn.gbm, testset, n.trees = churn.iter)
##Here the churn.predict output is neither classification nor probability. I guess it's log(probability).
str(churn.predict)
##Next, you can plot the ROC curve and get the best cut off that will have maximum accuracy:
library(pROC)
churn.roc = roc(testset$churn, churn.predict)
plot(churn.roc)
##You can retrieve the best cut off with the coords function and use this cut off to obtain the predicted label:
coords(churn.roc, "best")
churn.predict.class = ifelse(churn.predict > coords(churn.roc, "best")["threshold"], "yes", "no")
##Lastly, we can obtain the classification table from the predicted results:
table(testset$churn, churn.predict.class)


# Calculating the Margins of a Classifier
##A margin is a measure of the certainty of classification.
##We first prepare the data we need.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
set.seed(2)
churn.bagging = bagging(churn ~ ., data = trainset, mfinal = 10)
churn.predbagging = predict.bagging(churn.bagging, testset)
churn.boost = boosting(churn ~ ., data = trainset, mfinal = 10, coeflearn = "Freund", boos = F, control = rpart.control(maxdepth = 3))
churn.boost.pred = predict.boosting(churn.boost, testset)
##First, use the margins function to calculate the margins of the boosting classifiers:
boost.margins = margins(churn.boost, trainset)
boost.pred.margins = margins(churn.boost.pred, testset)
##We can then use the plot function to plot a marginal cumulative distribution graph of the boosting classifiers:
plot(sort(boost.margins[[1]]),
     (1:length(boost.margins[[1]])) / length(boost.margins[[1]]),
     type = "l", xlim = c(-1,1), main ="Boosting: Margin cumulative distribution graph", 
     xlab = "margin", ylab = "% observations", col = "blue")
lines(sort(boost.pred.margins[[1]]),
      (1:length(boost.pred.margins[[1]])) / length(boost.pred.margins[[1]]),
      type = "l", col = "green")
abline(v = 0, col = "red", lty = 2)
##We can then calculate the percentage of negative margin matches training errors and the percentage of negative margin matches test errors:
boosting.training.margin = table(boost.margins[[1]] > 0)
boosting.negative.training = as.numeric(boosting.training.margin[1] / boosting.training.margin[2])
boosting.negative.training

boosting.testing.margin = table(boost.pred.margins[[1]] > 0)
boosting.negative.testing = as.numeric(boosting.testing.margin[1] / boosting.testing.margin[2])
boosting.negative.testing
##Use the margins function to calculate the margins of the bagging classifiers:
bagging.margins = margins(churn.bagging, trainset)
bagging.pred.margins = margins(churn.predbagging, testset)
##We can then use the plot function to plot a marginal cumulative distribution graph of the bagging classifiers:
plot(sort(bagging.margins[[1]]),
     (1:length(bagging.margins[[1]])) / length(bagging.margins[[1]]),
     type = "l", xlim = c(-1,1), main = "Bagging: Margin cumulative distribution graph", 
     xlab = "margin", ylab = "% observations", col = "blue")
lines(sort(bagging.pred.margins[[1]]),
      (1:length(bagging.pred.margins[[1]])) / length(bagging.pred.margins[[1]]), 
      type="l", col = "green")
abline(v = 0, col = "red", lty=2)
##We can then compute the percentage of negative margin matches training errors and the percentage of negative margin matches test errors:
bagging.training.margin = table(bagging.margins[[1]] > 0)
bagging.negative.training = as.numeric(bagging.training.margin[1] / bagging.training.margin[2])
bagging.negative.training

bagging.testing.margin = table(bagging.pred.margins[[1]] > 0)
bagging.negative.testing = as.numeric(bagging.testing.margin[1] / bagging.testing.margin[2])
bagging.negative.testing
##For explanation about how margin works, please refer to the book to find more informations.


# Calculating the Error Evaluation of the Ensemble Method
##The adabag package provides the errorevol function for a user to estimate the ensemble method errors in accordance with the number of iterations.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(adabag)
churn.bagging = bagging(churn ~ ., data = trainset, mfinal = 10)
churn.boost = boosting(churn ~ ., data = trainset, mfinal = 10, coeflearn = "Freund", boos = F, control = rpart.control(maxdepth = 3))
##First, use the errorevol function to calculate the error evolution of the boosting classifiers:
boosting.evol.train = errorevol(churn.boost, trainset)
boosting.evol.test = errorevol(churn.boost, testset)
##Then, we draw a plot:
plot(boosting.evol.test$error, type = "l", ylim = c(0, 1),
     main = "Boosting error versus number of trees", xlab = "Iterations",
     ylab = "Error", col = "red", lwd = 2)
lines(boosting.evol.train$error, cex = 0.5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2, lwd = 2)
##The trend of the error rate can help measure how fast the errors reduce, while the number of iterations increases. 
##In addition to this, the graphs may show whether the model is over-fitted.
##---
##We then use the errorevol function to calculate the error evolution of the bagging claddifiers:
bagging.evol.train = errorevol(churn.bagging, trainset)
bagging.evol.test = errorevol(churn.bagging, testset)
plot(bagging.evol.test$error, type = "l", ylim = c(0,1),
     main = "Baggging error versus number of trees", xlab = "Iterations",
     ylab = "Error", col = "red", lwd = 2)
lines(bagging.evol.train$error, cex = 0.5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test","train"), col = c("red", "blue"), lty = 1:2, lwd = 2)
##The trend of the error rate can help measure how fast the errors reduce, while the number of iterations increases. 
##In addition to this, the graphs may show whether the model is over-fitted.


# Introduction to Random Forest
##The purpose of random forest is to ensemble weak learners (for example, a single decision tree) into a strong learner.
##Process:
##1. Assume that we have a training set containing N samples with M features. The process first performs bootstrap sampling, which samples N cases at random, with the replacement as the training dataset of each single decision tree. 
##2. In each node, the process first randomly selects m variables (where m << M), and then finds the predictor variable that provides the best split among m variables.
##3. The process grows the full tree without pruning.
##4. we can get the prediction result by taking an average or weighted average (for regression) of an output or taking a majority vote (for classification).
##---
##A random forest uses two parameters: ntree (the number of trees) and mtry (the number of features used to find the best feature), while the bagging method only uses ntree as a parameter.
##Therefore, if we set mtry equal to the number of features within a training dataset, then the random forest is equal to the bagging method.
##---
##Advantages and Disadvantages:
##The main advantages of random forest are that it is easy to compute, it can efficiently process data, and it is fault tolerant to missing or unbalanced data.
##The main disadvantage of random forest is that it cannot predict the value beyond the range of a training dataset. Also, it is prone to over-fitting of noisy data.


# Classifying Data with Random Forest
##Random forest is another useful ensemble learning method that grows multiple decision trees during the training process. 
##Each decision tree will output its own prediction results corresponding to the input. 
##The forest will use the voting mechanism to select the most voted class as the prediction result.
library(C50)
data(churn)
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]
library(randomForest)
##We first fit the model with the training set
churn.rf = randomForest(churn ~ ., data = trainset, importance = T)
##We set importance = T, which will ensure that the importance of the predictor is assessed.
##There are some other parameters that can be added in like: ntree, mtry, etc.
churn.rf
plot(churn.rf)
##We can examine the importance of each variable
importance(churn.rf)
##For a better visualization, we can use the varImpPlot function to obtain the plot of variable importance:
varImpPlot(churn.rf)
##Besides, we can also use the margin function to calculate the margins and plot the margin cumulative distribution
margins.rf = margin(churn.rf)
plot(margins.rf)
##You can also use boxplot to visualize the margins of the random forest by class:
boxplot(margins.rf ~ trainset$churn, main = "Margins of Random Forest for churn dataset by class")
##Next, make predictions based on the fitted model and testing data
churn.prediction = predict(churn.rf, testset)
table(churn.prediction, testset$churn)

##Apart from the randomForest package, we can use party package to provide an implementation of random forest.
library(party)
##We can then use the cforest function to fit the classification model:
churn.cforest = cforest(churn ~ ., data = trainset, controls = cforest_unbiased(ntree = 100, mtry = 5))
churn.cforest
##We can make predictions based on the built model and the testing dataset:
churn.cforest.prediction = predict(churn.cforest, testset, OOB = TRUE, type = "response")
##Finally, obtain the classification table from the predicted labels and the labels of the testing dataset:
table(churn.cforest.prediction, testset$churn)


# Estimating the Prediction Errors of Different Classifiers
##We will now validate whether the ensemble model performs better than a single decision tree by comparing the performance of each method. 
##In order to compare the different classifiers, we can perform a 10-fold cross-validation on each classification method to estimate test errors using erroreset from the ipred package.
##---
##Warnings: the code at the book seems a little strange, so I'll leave it.














