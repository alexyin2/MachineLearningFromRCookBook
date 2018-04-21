#Getting a dataset for machine learning
#Reference: http://archive.ics.uci.edu/ml/. ;  This website is called UCI machine learning repository
#Here we use iris data as an example
read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), 
         header = F, col.names = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width","Species"))

#Another Reference: http://www.kdnuggets.com/datasets/index.html ; This website is called KDnuggets
