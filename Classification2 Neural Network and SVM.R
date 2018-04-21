##Most Research has shown that Support Vector Machine and Neural Networks are powerful classification tools
##But the problem is that is quite hard to interpret the process from input to output
##People called the phenomenon as a "blackbox"

# Support Vector Machine
##support vector machines first map input data into a high dimension feature space defined by the kernel function and then find the optimum hyperplane that separates the training data by the maximum margin. 
##In short, we can think of support vector machines as a linear algorithm in a high dimensional space.
##The basic idea behind Support Vector Machine is for binary classification.
##But we can also use SVM to predict continuous values
##Think of input data as x and y. If we have two input data points, then we have two-dimensional space.
##The hyperplane is a line that splits the input in space.
##So if there is a lot of space for speration, we should select a line that maximizes margin where the distance between the closest point and the line is at a minimum.
##The data used is as follows:
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
##We use library(e1071) here for SVM 
library(e1071)
##Train the support vector machine using the svm function with trainset as the input dataset and use churn as the classification category:
model = svm(churn ~ ., data = trainset, kernal = "radial", cost = 1, gamma = 1 / ncol(trainset))
##We can obtain overall information about the built model with summary:
summary(model)
##The advantage of using SVM is that it builds a hightly accurate model through an engineering problem-oriented kernal
##It makes use of the regularization term to avoid over-fitting
##It also doesn't suffer from local optimal and multicollinearity
##The main limitation of SVM is it's speed and size in training and testing time
##Therefore, it's not suitable or efficient enough to contrust classification models for data that is large in size
##---------------------------------
##We begin training a support vector machine using libsvm provided in the e1071 package
##Within the training function, we can specify the kernal function, cost, and gamma function:
##1. For kernal argument: the default value is radial; we can change it to linear, polynomial, radial basis, and sigmoid
##2. For gamma argument: the default value is (1 / data dimension), and it controls the shape of the separating hyperplane. Increasing gamma usually increases the number of support vectors
##3. For the cost: the default value is 1, which indicates the regularization term is constant and the larger the value, the smaller the margin is. 
##---------------------------------
##Another popular support vector machine tool is SVMLight. Unlike the e1071 package, which provides the full implementation of libsvm, the klaR package simply provides an interface to SVMLight

# Choosing the Cost of a Support Vector Machine
##Sometimes, we would like to allow some missclassifications while seperating categories
##The SVM model has a cost function, which controls training errors and margins
##For example, a small cost creates a large margin (a soft margin) and allows more misclassifications. On the other hand, a large cost creates a narrow margin (a hard margin) and permits fewer misclassifications
data("iris")
iris.subset = subset(iris, select = c("Sepal.Length", "Sepal.Width", "Species"), subset = Species %in% c("setosa", "virginica"))
plot(x = iris.subset$Sepal.Length, y = iris.subset$Sepal.Width, col = iris.subset$Species, pch = 19)
##Next, we train SVM based on iris.subset with the cost equal to 1:
svm.model = svm(Species ~ ., data = iris.subset, kernal = 'linear', cost = 1, scale = F)
plot(x = iris.subset$Sepal.Length, y = iris.subset$Sepal.Width, col = iris.subset$Species, pch = 19)
points(iris.subset[svm.model$index, c(1,2)], col = "blue", cex = 2)
##Lastly, we can add a separation line to the plot:
w = t(svm.model$coefs) %*% svm.model$SV
q = -svm.model$rho
plot(x = iris.subset$Sepal.Length, y = iris.subset$Sepal.Width, col = iris.subset$Species, pch = 19)
points(iris.subset[svm.model$index, c(1,2)], col = "blue", cex = 2)
abline(a = -q/w[1,2], b = -w[1,1]/w[1,2], col = "red", lty = 5)
##We next train a model with a cost = 10,000
svm.model = svm(Species ~ ., data = iris.subset, type = 'C-classification', kernal = 'linear', cost = 10000, scale = F)
plot(x = iris.subset$Sepal.Length, y = iris.subset$Sepal.Width, col = iris.subset$Species, pch = 19)
points(iris.subset[svm.model$index, c(1,2)], col = "blue", cex = 2)
w = t(svm.model$coefs) %*% svm.model$SV
q = -svm.model$rho
abline(a = -q/w[1,2], b = -w[1,1]/w[1,2], col = "red", lty = 5)
##A small cost creates a large margin (a soft margin) and allows more misclassifications. On the other hand, a large cost creates a narrow margin (a hard margin) and permits fewer misclassifications
##-----------------------------
##The abline seems not working when was running the code, may need to find a solution
##------------------------------

# Visualizing an SVM fit
data("iris")
library(C50)
data("churn")
##Use SVM to train the support vector machine based on the iris dataset and use the plot function to visualize the fitted model:
model.iris = svm(Species ~ ., data = iris)
plot(model.iris, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))
##Visualize the SVM fit object, model, using the plot function with the dimensions of total_day_minutes and total_intl_charge:
model = svm(churn ~ ., data = trainset, kernal = "radial", cost = 1, gamma = 1 / ncol(trainset))
plot(model, trainset, total_day_minutes ~ total_intl_charge)
##For more information about the explanation to the plot, please look at the book


# Predicting based on a model trained by SVM
svm.pred = predict(model, testset[, !names(testset) %in% c("churn")])
##Perform a table
svm.table = table(svm.pred, testset$churn) ; svm.table
##Next, we can use classAgreement to calculate coefficients compared to the classification agreement:
library(e1071)
classAgreement(svm.table)
##The diag coefficient represents the percentage of data points in the main diagonal of the classification table
##kappa refers to diag, which is corrected for an agreement by a change (the probability of random agreements)
##rand represents the rand index, which measures the similarity between two data clusters
##crand indicates the rand index, which is adjusted for the chance grouping of elements
##---
##We can use confusionMatrix to measure the predict performance based on the classification table
##We can use SVM to predict continuous values. In other words, one can use SVM to perform regression analysis.
library(caret)
confusionMatrix(svm.table)

# Predicting continuous values by SVM
library(car)
data("Quartet")
model.regression = svm(Quartet$y1 ~ Quartet$x, type = "eps-regression")
##Use the predict function to obtain prediction results:
predict.y = predict(model.regression, Quartet$x )
##Plot the predicted points as squares and the training data points as circles on the same plot:
plot(Quartet$y1 ~ Quartet$x, pch = 19)
points(predict.y ~ Quartet$x, pch = 15, col = "red")

# Tuning a Support Vector Machine
##Besides using different feature sets and the kernal function in SVM, we can also adjust gamma and the cost
##SVM provides a tuning function tune.svm() to help test for the best cost and gamma value
##First, tune the support vector machine using tune.svm:
tuned = tune.svm(churn ~ ., data = trainset, gamma = 10^(-6:-1), cost = 10^(1:2))
##Next, we can use the summary function to obtain the testing result
summary(tuned)
##After retrieving the best performance parameter from tuning the result, you can retrain the support vector machine with the best performance parameter:
model.tuned = svm(churn ~ ., data = trainset, gamma = tuned$best.parameters$gamma, cost = tuned$best.parameters$cost)
summary(model.tuned)
##Then, we can use the predict function to predict labels based on the fitted SVM:
svm.tuned.pred = predict(model.tuned, testset[, !names(testset) %in% c("churn")])
svm.tuned.table = table(svm.tuned.pred, testset$churn) ; svm.tuned.table
##Generate a class agreement to measure the performance:
library(e1071)
classAgreement(svm.tuned.table)
##For more information about the explanation of the parameters in classAgreement, please find the previous code
##At last, use a confusion matrix to measure the performance of the retrained model:
confusionMatrix(svm.tuned.table)

# Neural Network
##There are three types of neurons within the network: input neurons, hidden neurons, and output neurons.
##In the network, neurons are connected; the connection strength between neurons is called weight. If the weight is greater than zero, it is in an excitation status. Otherwise, it is in an inhibition status.
##---
##The advantage of Neural Network:
##1. It can detect nonlinear relationships between the dependent and independent variable
##2. One can efficiently train large datasets using the parallel architecture.
##3. It is a nonparametric model so that one can eliminate errors in the estimation of parameters.
##The disadvantage of Neural Network:
##1. It often converges on the local minimum, rather than the global minimum.
##2. It might over-fit when the training process goes on for too long.

library(neuralnet)
##We first sploit iris data into trainset and testset
data(iris)
ind = sample(c(1:2), nrow(iris), replace = T, prob = c(0.7, 0.3))
trainset = iris[ind == 1,]
testset = iris[ind == 2, ]
##Add the columns versicolor, setosa, and virginica based on the name matched value in the Species column:
trainset$setosa = (trainset$Species == "setosa")
trainset$virginica = (trainset$Species == "virginica")
trainset$versicolor = (trainset$Species == "versicolor")
##Next, we train the neural network with the neuralnet function with 'three' hidden neurons in each layer
network = neuralnet(versicolor + virginica + setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                    data = trainset, hidden = 3)
##we can view the summary information by accessing the result.matrix attribute of the built neural network model:
network$result.matrix
##From the output above, we can see how many steps the training process need until all the partial derivatives of the error function were lower than 0.01.(specified in the threshold)
##The error refers to the likelihood of calculating Akaike Information Criterion (AIC)
##---
##Lastly, you can view the generalized weight by accessing it in the network:
head(network$generalized.weights[[1]])
##Look at the book to gain more information

#Visualizing a Neural Network trained by neuralnet
network = neuralnet(versicolor + virginica + setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                    data = trainset, hidden = 3)

plot(network)
##The plot includes the estimated weight, intercepts and basic information about the training process. 
##At the bottom of the figure, one can find the overall error and number of steps required to converge.
#---
##Furthermore, you can use gwplot to visualize the generalized weights:
par(mfrow=c(2,2))
library(neuralnet)
gwplot(network,selected.covariate="Petal.Width")
gwplot(network,selected.covariate="Sepal.Width")
gwplot(network,selected.covariate="Petal.Length")
gwplot(network,selected.covariate="Petal.Width")
##The four plots in the preceding figure display the four covariates: Petal.Width, Sepal.Width, Petal.Length, and Petal.Width, in regard to the versicolor response
##If all the generalized weights are close to zero on the plot, it means the covariate has little effect. 
##However, if the overall variance is greater than one, it means the covariate has a nonlinear effect.

# Predicting labels based on model trainde by Neural Network
##We first generate a prediction probability matrix based on a trained neural network and the testset:
net.predict = compute(network, testset[-5])$net.result
##Then, obtain other possible labels by finding the column with the greatest probability:
net.prediction = c("setosa", "virginica", "versicolor")[apply(net.predict, 1, which.max)]
##On the above function, to convert the probability matrix to class labels, we use the which.max function to determine the class label by selecting the column with the maximum probability within the row
##Generate a classification table based on the predicted labels and the labels of the testing dataset:
predict.table = table(testset$Species, net.prediction) ; predict.table
##Finally, use confusionMatrix to measure the prediction performance:
library(caret)
confusionMatrix(predict.table)

# Training a neural network with nnet
library(nnet)
data("iris")
set.seed(2)
ind = sample(c(1:2), nrow(iris), replace = T, prob = c(0.7,0.3))
trainset = iris[ind == 1, ]
testset = iris[ind == 2, ]
iris.nn = nnet(Species ~ ., data = trainset, size = 2, rang = 0.1, decay = 5e-4, maxit = 200)
##Explain of parameters:
##size: the number of hidden units
##rang: the initial random weight
##decay: the parameter for weight decay
##maxit: the maximum iteration
##we set maxit to 200, the training process repeatedly runs till the value of the fitting criterion plus the decay term converge.
summary(iris.nn)
##summary(iris.nn) shows that the model is built with 4-2-3 networks with 19 weights.
##Also, the model shows a list of weight transitions from one node to another at the bottom of the printed message.

# Predicting labels based on a model trained by nnet
iris.predict = predict(iris.nn, testset, type = "class")
nn.table = table(testset$Species, iris.predict)
nn.table
library(caret)
confusionMatrix(nn.table)

##For the predict function, if the type argument to class is not specified,it will, by default, generate a probability matrix as a prediction result. 
##This is very similar to net.result generated from the compute function within the neuralnet package
head(predict(iris.nn, testset))

