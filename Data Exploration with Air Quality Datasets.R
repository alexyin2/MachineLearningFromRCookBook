data("airquality")
View(airquality)
str(airquality)
mydata = airquality
mydata$Month = factor(mydata$Month)

#Detecting Missing values
##See the percentage of missing values in Ozone
sum(is.na(mydata$Ozone)) / length(mydata$Ozone)
##See all percentage of missing values in airquality
sapply(mydata, function(df){sum(is.na(df)==TRUE)/length(df)})

#Missing Value Visualization
library(Amelia)
missmap(mydata, main="Missing Map")

#Impute missing values
#Here,we want to replace the NA values with the most repeated observation.
##Use the table command to find the frequency of the observations
table((factor(mydata$Ozone)),useNA="always")
##The max frequency is 6, which belongs the the variable 23
max(table((factor(mydata$Ozone))))
##We replace missing values in Ozone with 23. Notice that it isn't a good way
mydata$Ozone[which(is.na(mydata$Ozone))]=23
##Doing the same on variable Solar.R
##The max frequency is 4, which belongs the the variable 259
max(table((factor(mydata$Solar.R))))
##We replace missing values in Solar.R with 259ß. Notice that it isn't a good way
mydata$Solar.R[which(is.na(mydata$Solar.R))]=259

#Exploring and Visualizing data
barplot(table(mydata$Ozone), main="Ozone Observations",xlab="O bservations", ylab="Frequency")
barplot(table(mydata$Temp), main="Temperature Observations",xlab="Temprature", ylab="Frequency")
hist(mydata$Temp, main="Temperature", xlab = " Temperature ")
hist(mydata$Temp,  main="Temperature", xlab = " Temperature ",prob=TRUE)
boxplot(mydata)
boxplot(mydata$Temp ~ mydata$Month, main="Month Wise Temperature",xlab="Month", ylab="Temperature")
pairs(mydata[1:4])

#Predicting values from datasets
##First we use corrplot to draw a correlation matrix 
##Use ?corrplot to see more different methods
library(corrplot)
mydata$Month = airquality$Month
par(mfrow=c(1,1))
corrplot(cor(mydata),method="number")
corrplot(cor(mydata),method="circle")
##Second, we use Simple Regression to build our model
reg = lm(Temp~Ozone,data=mydata)
summary(reg)