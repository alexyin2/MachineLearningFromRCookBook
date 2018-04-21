# Introduction of Classification
##classification is used to identify the category of a given observation

# Removing Unused Columns and Preparing Training Data and Testing Data
library(C50)
data(churn)
str(churnTrain)
##We first remove state, area_code, and account_lenghts, which are not appropriate for classification
churnTrain = churnTrain[, !names(churnTrain) %in% c("state", "area_code", "account_length") ]
##The above is a really convenient way to remove columns in a dataframe
## x %in% Y means to make logic judgement if Y is in the set X
##Next, we will introduce the way to split data into training set and testing set
##We will use churnTrain data as an example:
set.seed(2)
ind = sample(c(1,2), size = nrow(churnTrain), replace = T, prob = c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2, ]

# Recursive Partitioning and Regression Trees
##The advantage of decision tree is that we don't have to worry whether the data is linear separable.
##As for the disadvantage of using the decision tree, it is that it tends to be biased and over-fitted.
##However, you can conquer the bias problem through the use of a conditional inference tree, and solve the problem of over-fitting through a random forest method or tree pruning.
##The classification process starts from the root node of the tree; 
##at each node, the process will check whether the input value should recursively continue to the right or left sub-branch according to the split condition, 
##and stops when meeting any leaf (terminal) nodes of the decision tree.
##We first make sure we completed the codes above
library(rpart)
churn.rp = rpart(churn ~ ., data = trainset)
##In rpart() there are some parameters that may be useful:
##minsplit: the minimum number of observations that must exist in a node in order for a split to be attempted
##cp: complexity parameter
churn.rp
##Explanation about meanings in churn.rp:
##n: sample size ; loss: the missclassification loss ; 
##yval: classified membership (no or yes, in this case) ; yprob = the probabilities of two classes(no, yes)
##root: the first split of the Decision Tree
##terminal node: also calles leaf node, which is the split that has no further classification
##Use printcp() to examine the complexity parameter
printcp(churn.rp)
##Explanation about meanings in printcp(churn.rp):
##CP: complexity parameter, which serves as a penalty to control the size of the tree. The greater the CP value, the fewer the number of splits (nsplit) there are.
##rel error: the average deviance of the current tree devided by the average deviance of the null tree
##xerror: the relative error estimated by a 10-fold classification
##xstd: standard error of the relative error
##Next, use the plotcp function to plot the cost complexity parameters:
plotcp(churn.rp)
##Explanation of the plot:
##y-axis: the relatvie error, detail explanations can be found above ; The dotted line indicates the upper limit of a standard deviation
##In this plot, we can determine that minimum cross-validation error occurs when the tree is at a size of 12
##Lastly, use the summary function to examine the built model:
summary(churn.rp)
##Explanations of summary(churn.cp):
##We first can see a matrix showing that the cp minimum is at 0.01 and the respond nsplit = 12, which gives the same result in our plot.
##Last, we can see variable importance, which can help us indentify the most important variable (the numbers of each variable importance are summing up to 100)
sample1 = rpart(churn~., data = trainset, minsplit = 1, cp = 1e-4)
sample1
printcp(sample1)
plotcp(sample1)
##The above is an example when we adjust the parameters in rpart()
##minisplit: the minimum number of observations that must exist in a node in order for a split to be attempted
##cp: when the functions stops until it acheives the cp we set ; here 1e-4 means 10^(-4)
##sample1's nsplit = 158 , which is really overfitting in this case

# Visualizing a Recursive Partioning and Regression Tree
library(rpart)
churn.rp = rpart(churn ~ ., data = trainset)
plot(churn.rp, margin = 0.1)
text(churn.rp, all = T, use.n = T)
##We can see that the plot shows that nsplit = 12, which gives the same result as the previous code
##We can also specify the uniform, branch, and margin parameter to adjust the layout
plot(churn.rp, uniform = T, branch = 0.6, margin = 0.1)
text(churn.rp, all = T, use.n = T)
##Explanation of parameters:
##margin is a parameter to add extra white space around the border to prevent the displayed text being truncated
##In order to add extra information on the tree plot, we set the parameter as all = TRUE to add a label to all the nodes
##use.n = TRUE add extra information, which shows that the actual number of observations falls into two different categories (no and yes).
##Use ?plot.rpart to gain more inforamations

# Measuring the Prediction Performance of a Recursive Partioning and Regression Tree
library(rpart)
churn.rp = rpart(churn ~ ., data = trainset)
predictions = predict(churn.rp, testset, type = "class")
table(testset$churn, predictions)
##We can further generate a confusion matrix using the confusionMatrix function provided in the caret package:
library(caret)
confusionMatrix(table(predictions, testset$churn))
##Gain information about the meanings in confusionMatrix: http://en.wikipedia.org/wiki/Confusion_matrix

# Pruning a Recursive Partioning and Regression Tree
##Find the minimum cross-validation error of the classification tree model
printcp(churn.rp)
names(churn.rp)
min(churn.rp$cptable[,"xerror"])
##Locate the record with the minimum cross-validation errors:
which.min(churn.rp$cptable[,"xerror"])
##the output is 7
##Get the cost complexity parameter of the record with the minimum cross- validation errors:
churn.cp = churn.rp$cptable[7, "CP"]
churn.cp
##Prune the tree by setting the cp parameter to the CP value of the record with minimum cross-validation errors:
prune.tree = prune(churn.rp, cp = churn.cp)
##Visualize the classification tree by using the plot and text function:
plot(prune.tree, margin = 0.1)
text(prune.tree, all = T, use.n = T)
##Next, we can generate a classification table based on the pruned classification tree model:
predictions = predict(prune.tree, testset, type = "class")
table(testset$churn, predictions)
##Lastly, we can generate a confusion matrix based on the classification table:
confusionMatrix(table(predictions, testset$churn))
##The result shows that the accuracy (0.9411) is slightly lower than the original model (0.942), and also suggests that the pruned tree may not perform better than the original classification tree as we have pruned some split conditions 
##still, one should examine the change in sensitivity and specificity. However, the pruned tree model is more robust as it removes some split conditions that may lead to over-fitting.

# Handling Missing Data and Split and Surrogate Variables
##Missing data can be a curse for ananlysis and prediction.
##One simple way is to discard the data with missing variables.
##But the action is only suggested when missing values are less than 5 percent of the total dataset.
#Here we're using package mice ,vim and randomForest to impute missing values
library(mice)
library(randomForest)
library(VIM)
##Find the minimum cross-validation error of the classification tree model
t = data.frame(x = c(1:100), y = c(1:100))
t$x[sample(1:100, 10, replace = F)] = NA
t$y[sample(1:100, 10, replace = F)] = NA
##We use aggr() to visualize the missing values 
aggr(t)
##Adding more parameters to gain more informatino
aggr(t, prop = T, numbers = T)
##We use matrixplot() for another visualized plot
matrixplot(t)

# Conditional Inference Tree
##Similar to traditional decision trees, conditional inference trees also recursively partition the data by performing a univariate split on the dependent variable
##However, what makes conditional inference trees different from traditional decision trees is that conditional inference trees adapt the significance test procedures to select variables rather than selecting variables by maximizing information measures
library(party)
ctree.model = ctree(churn ~ ., data = trainset)
ctree.model

# Controlling Parameters in Conditional Inference Tree
ctree.model = ctree(churn ~ ., data = trainset, 
                    controls = ctree_control(testtype = "MonteCarlo", mincriterion = 0.90, minbucket = 15))
ctree.model
##We said we are going to use the MonteCarlo simulation with minimum weight on node at 15 and mincriterion at 0.90.
##We can also use Bonferroni, Univariate, and Teststatistic in place of MonteCarlo.
##Use ?ctree and ?ctree_control for more information

# Visualizing a Conditional Inference Tree
plot(ctree.model)
##To obtain a simple conditional inference tree, one can reduce the built model with less input features and redraw the classification tree:
daycharge.model = ctree(churn ~ total_day_charge, data = trainset)
plot(daycharge.model)
##The output figure reveals that every intermediate node shows the dependent variable name and the p-value.
##The terminal nodes show the number of categorized observations, n, and the probability of a class label of either 0 or 1.
##Now we take plot(daycharge,model) as an example and explain:
##We can find out when total_day_charge is above 48.18, the lighter gray area is larger than the darker gray area in node 9.
##This indicates that the customer with a total_day_charge over 48.18 has a greater likelihood to churn(labe = yes)

#Measuring the Prediction Performace of a Conditional Inference Tree
library(party)
ctree.model = ctree(churn ~ ., data = trainset)
ctree.predict = predict(ctree.model, testset, type = "response")
##The above function predict() can have three types: response, prob, and node
table(ctree.predict, testset$churn)
##Furthermore, we can use confusionMatrix from the caret package to generate the performance measurements of the prediction result:
library(caret)
confusionMatrix(table(ctree.predict, testset$churn))
##We can also use the treeresponse function, which will tell us the list of class probabilities:
tr = treeresponse(ctree.model, newdata = testset[1:5,])
tr
##If we choose type = "prob" in the functino predict() than we will get the same result as tr

# K-Nearest Neighbor Classifier (KNN)
##K-nearest neighbor (knn) is a nonparametric lazy learning method.
##The advantages of Knn:
##1. The cost of the learning process is zero
##2. It is nonparametric, which means we don't have to make the assumption of data distribution
##3. We can classify any data if we can find similliar measures of given instances
##The disadvantages of knn:
##1. It is hard to interpret the classified result
##2. It is an expensive computation for large datasets
##3. The performance relies on the number of dimensions. Therefore, for a high dimension problem, we should reduce the dimension first to increase the process performance
##4. If variables are non-numeric, we have to transform the datatype into numeric
##To choose a proper k-value, one can count on cross-validation.
library(class)
##Replace yes and no of the voice_mail_plan and international_plan attributes in both the training dataset and testing dataset to 1 and 0:
levels(trainset$international_plan)
levels(trainset$voice_mail_plan)
levels(trainset$international_plan) = list("0" = "no", "1" = "yes")
levels(trainset$voice_mail_plan) = list("0" = "no", "1" = "yes")
levels(testset$international_plan) = list("0" = "no", "1" = "yes")
levels(testset$voice_mail_plan) = list("0" = "no", "1" = "yes")
##Use Knn classification method on the trainset and testset
churn.knn = knn(train = trainset[, !names(trainset) %in% c("churn")],
               test = testset[, !names(testset) %in% c("churn")], 
               trainset$churn,
               k = 3)
## x %in% Y means to make logic judgement if Y is in the set X
##We can use summary(churn.kn) to retrieve the number of predicted labels
summary(churn.knn)
## Next, we can generate the classification matrix using the table function
table(testset$churn, churn.knn)
##Lastly, we can generate a confusion matrix by using the confusionMatrix function:
library(caret)
confusionMatrix(table(testset$churn, churn.knn))
##There is also another package called kknn, which provides a weighted knn classification

# Logistic Regression
fit = glm(churn~., data = trainset, family = binomial)
summary(fit)
##The above summary shows that many variables are not significant, so we deleted them and run another model
fit = glm(churn ~ international_plan + voice_mail_plan + total_intl_calls + number_customer_service_calls,
          data = trainset, family=binomial)
summary(fit)
pred = predict(fit, testset, type = "response")
##the type her should be: link, response, or link
Class = pred > 0.5
##We can decide the judgement probability by our own
summary(Class)
tb = table(testset$churn, Class)
tb
##The advantage of Logistic Regression is that it's fast to update the model
##The disadvantage is that the algorith suffer from multicollinearity, therefore, the explanatory variables must be linear independent

# Naïve Bayes Classifier
##More introduction: http://cpmarkchang.logdown.com/posts/193470-natural-language-processing-naive-bayes-classifier
##The Naïve Bayes classifier is also a probability-based classifier, which is based on applying the Bayes theorem with a strong independent assumption
library(e1071)
classifier = naiveBayes(trainset[, !names(trainset) %in% c("churn")], trainset$churn)
##x %in% Y means to make logic judgement if Y is in the set X
classifier
##We can generate a classification table for the testing data
bayes.table = table(predict(classifier, testset[, !names(testset) %in% c("churn")]), testset$churn)
bayes.table
##Next, we can generate a confusionMatrix from the classification table
library(caret)
confusionMatrix(bayes.table)
##It is suitable when the training set is relative small, and may contain some noisy and missing data.
## The drawbacks of Naïve Bayes are that it assumes that all features are independent and equally important, which is very unlikely in real-world cases

# For more comparison, look at the picuture: Comparison1 Table Comparison
