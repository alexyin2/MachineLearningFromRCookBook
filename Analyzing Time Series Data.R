#Let's think of the share price of some company in the range of 2,500 to 4,000 from 2011 to be recorded monthly. 
#Use ts() to perform a time series data
my_vector = sample(2500:4000,72,replace = T)
my_series = ts(my_vector, start = c(2011,1), end = c(2016,12), frequency = 12)
class(my_series)
my_series

#Ploting and forecasting time series data
plot(my_series)
plot(AirPassengers)
cycle(AirPassengers)
##Here the aggregate() splits the data into years
aggregate(AirPassengers)
plot(aggregate(AirPassengers))
boxplot(AirPassengers~cycle(AirPassengers))
##Installing forecast package
library(forecast)
##Here forecast(my_series, 4) means to forecast the value for the next 4 months
forecast(my_series, 4)
##Holt-Winters is used for exponential smoothing, which is covered later in this chapter
f = HoltWinters(my_series)
forecast(f, 4)

#Extracting, subsetting, merging, filling, and padding
library(xts)
first(my_series)
first(my_series, "3 months")
first(my_series, "3 years")
my_series1 = ts(my_vector, start=c(2011,1), end=c(2013,12), frequency =4)
my_series1[c(2,5,10,13)]=NA
my_series1

#Successive differences and moving averages
##A successive difference means the difference between the successive observations.
##For time series y, the successive difference will be (y2-y1), (y3-y2), (y4-y3)

#1. Moving-average Smoothing
library(forecast)
##ma() makes a moving-average smothing, which is used to understand the direction of the current trend.
##The moving average will smooth the data if there is a variation in consecutive points.
##It will smoothen the curve on the base of average. In addition, it may eliminate some data to avoid randomness.
ma(my_series, order = 10)
plot(ma(my_series, order = 10))
##Make a comparison with plot(my_series)

#2. Exponential Smoothing
##In moving-average smoothing, all observations are weighted equally.
##But in Exponential Smoothing, the weights are assigned in exponentially decreasing order as the observation gets old.
##This ensures that the recent or latest observations are given more weightage thus we can forecast on the basis of recent observations
library(forecast)
t = ets(AirPassengers) ; t
plot(t)
##Make a comparison with plot(AirPassengers)
val = forecast(t)
plot(val)

#Ploting the autocorelation function
##The acf() function will plot the correlation between all pairs of data points with lagged values.
##The plot will have two horizontal blue dashed lines at -0.2 and 0.2, representing the upper and lower bounds
##If auto-correlation coefficients are close to zero, this means that there is no relationship
##E.g. 1
lag.plot(AirPassengers, lags = 10)
acf(AirPassengers)
##E.g. 2
sales = sample(400:10000, 72, replace = T)
my_newseries = ts(sales, start = c(2011,1), end = c(2016,12), frequency = 12)
my_newseries
lag.plot(my_newseries, lags = 10)
acf(my_newseries)
##Here we can see that there is some autocorrelations AirPassengers data ; 
##whereas there is little autocorrelation in my_newseries data

##The parameter lag here means the time displacement. 
##E.g., We have a time series Y1, Y2, .....Y7, then Y2 and Y5 have lag 3 (5 - 2 = 3) 
##So here we take lag 1 in the plot as an example. The point in the plot refers to (x = Yi, y = Yi-1)
##So if there are patterns in the plot, then it may imply that theres an autocorrelation in the data

##中文版：我們以 Lag 1 舉例，如果圖中呈現一正相關，代表yi愈大，則yi的前一個數值，也就是yi-1也愈大
##中文版：我們以 Lag 2 舉例，如果圖中呈現一正相關，代表yi愈大，則yi的前二個數值，也就是yi-2也愈大
