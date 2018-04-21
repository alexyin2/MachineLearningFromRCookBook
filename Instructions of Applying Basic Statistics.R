#Use data() to check all the datasets in R
data()

#Call the data we want
data(iris)

#Use sapply to apply the function on the input data
# iris[1:4] ; iris[,c(1:4)]  These two are the same
sapply(iris[1:4],mean,na.rm=T)

#Use summary()
summary(iris)

#Use cov() to compute the covariance of each attribute pair
cov(iris[,c(1:4)])

#Use t.test() to perform a t-test
t.test(iris$Petal.Width[iris$Species=='setosa'],
       iris$Petal.Width[iris$Species=='versicolor'])

#Use cor()
cor.test(iris$Sepal.Length, iris$Sepal.Width)

#Use aggregate()
aggregate(x=iris[,1:4],by=list(iris$Species),FUN=mean)

#Use library reshape
library(reshape)
iris.melt = melt(iris,id='Species')
cast(Species~variable, data = iris.melt, subset = Species %in% c('setosa','versicolor'), margins = 'grand_row')

#Use pie() to draw a pie chart
table.iris = table(iris$Species)
pie(table.iris)

#Use hist() to draw a histogram
hist(iris$Sepal.Length)

#Use boxplot() to draw a boxplot
boxplot(Petal.Width~Species,data = iris)

#Use plot() to draw a plot
plot(x = iris$Petal.Length, y = iris$Petal.Width, col = iris$Species)

#Use pairs() to draw scatter plot
pairs(iris[1:4], main = "Iris data", pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
##bg: The color to be used for the background of the device region ; pch: different kinds of shapes in the plot
