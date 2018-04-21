# Introduction to Clustering
##Unlike supervised learning methods (for example, classification and regression) covered in the previous chapters, a clustering analysis does not use any label information, but simply uses the similarity between data features to group them into clusters.
##The four most common types of clustering methods are hierarchical clustering, k-means clustering, model-based clustering, and density-based clustering:
##1.Hierarchical Clustering: This creates a hierarchical of cluster, and presents the hierarchy in a dendrogram.
##This method does not require the number of cluster to be specified at the beginning.
##2. k-means clustering: This is also referred to as flat clustering. Unlike hierarchical clustering, it does not create a hierarchy of clusters, and it requires the number of clusters as an input.
##However, its performance is faster than hierarchical clustering.
##3. Model-based clustering: Both hierarchical clustering and k-means clustering use a heuristic approach to construct clusters, and do not rely on a formal model. 
##Model-based clustering assumes a data model and applies an EM algorithm to find the most likely model components and the number of clusters.
##4.Density based clustering: This constructs clusters in relation to the density measurement. Clusters in this method have a higher density than the remainder of the dataset.
##---
##We will discuss how to validate clusters internally, using within clusters the sum of squares, average silhouette width, and externally, with ground truth.


# Clustering Data with Heirarchical Clustering
##Hierarchical clustering adopts either an agglomerative or divisive method to build a hierarchy of clusters.
##Generally, there are two approaches to building hierarchical clusters:
##1. Agglomerative hierarchical clustering:
##This is a bottom-up approach. Each observation starts in its own cluster. 
##We can then compute the similarity (or the distance) between each cluster and then merge the two most similar ones at each iteration until there is only one cluster left.
##2. Divisive hierarchical clustering:
##This is a top-down approach. All observations start in one cluster, and then we split the cluster into the two least dissimilar clusters recursively until there is one cluster for each observation
##---
##Regardless of which approach is adopted, both first use a distance similarity measure to combine or split clusters. 
##The recursive process continues until there is only one cluster left or you cannot split more clusters.
##Eventually, we can use a dendrogram to represent the hierarchy of clusters.
##---
##The files using here can be downloaded at https://github.com/ywchiu/ml_R_cookbook/tree/master/CH9
customer = read.csv("/Users/owner1/Desktop/R/Machine Learning of R Cookbook/Clustering Example Files/customer.csv")
str(customer)
summary(customer)
##We should normalize the customer data into the same scale:
customer = scale(customer[,-1])
##scale is a function that normalize the data we selected.
##1. We can then use agglomerative hierarchical clustering to cluster the customer data:
hc = hclust(dist(customer, method = "euclidean"), method = "ward.D2")
##From the above function, We use the Euclidean distance as distance metrics, and use Ward's minimum variance method to perform agglomerative clustering.
##Ward method: This refers to the sum of the squared distance from each point to the mean of the merged clusters
hc
##Lastly, you can use the plot function to plot the dendrogram:
plot(hc, hang = -0.01, cex = 0.7)
##In plot(hc, hang = -0.01, cex = 0.7), We specify hang to display labels at the bottom of the dendrogram, and use cex to shrink the label to 70 percent of the normal size.
##---
##Additionally, you can use the single method to perform hierarchical clustering and see how the generated dendrogram differs from the previous:
hc2 = hclust(dist(customer), method = "single")
plot(hc2, hang = -0.01, cex = 0.7)
##2. Besides agglomerative hierarchical clustering; if we would like to perform divisive hierarchical clustering, we can use the diana function:
library(cluster)
dv = diana(customer, metric = "euclidean")
summary(dv)
plot(dv)


# Cutting Trees into Clusters
##From the previous learning, we know there are two ways for heirarchical clustering.
##But we haven't group the data into clusters yet, we can't so that we're going to have n/2 clusters since it's too much.
##So now, we're going to determine how many clusters are within the dendrogram and cut the dendrogram at a certain tree height to separate the data into different groups
##In this recipe, we will demonstrate how to use the cutree function to separate the data into a given number of clusters.
customer = scale(customer[,-1])
hc = hclust(dist(customer, method = "euclidean"), method = "ward.D2")
fit = cutree(hc, k = 4)
table(fit)
##In table(fit), we can see haw many datas are in each cluster
##---
##We can also visualize how data is clustered with a red rectangle border
plot(hc)
rect.hclust(hc, k = 4, border = "red")
##Besides drawing rectangles around all hierarchical clusters, you can place a red rectangle around a certain cluster:
plot(hc)
rect.hclust(hc, k = 4, which = 2, border = "red")


# Clustering Data with the K-means Method
##K-means requires the number of clusters to be determined first.
##k-means clustering is much faster than hierarchical clustering as the construction of a hierarchical tree is very time consuming.
customer = read.csv("/Users/owner1/Desktop/R/Machine Learning of R Cookbook/Clustering Example Files/customer.csv")
customer = scale(customer[,-1])
set.seed(2)
fit = kmeans(customer, 4)
fit
##We can then inspect the center of each cluster using barplot:
barplot(t(fit$centers), beside = T, xlab = "cluster", ylab = "value")
##We can draw the centers of each cluster in a bar plot, which will provide more details on how each attribute affects the clustering.
##---
##Lastly, we can draw a scatter plot of the data and color the points according to the clusters:
plot(customer, col = fit$cluster)


# Drawing a Bivariate Cluster Plot
##In the previous recipe, we employed the k-means method to fit data into clusters. 
##However, if there are more than two variables, it is impossible to display how data is clustered in two dimensions. 
##Therefore, you can use a bivariate cluster plot to first reduce variables into two components, and then use components, such as axis and circle, as clusters to show how data is clustered.
library(cluster)
##We can draw a bivariate cluster plot
clusplot(customer, fit$cluster, color = TRUE, shade = TRUE)
##We can also zoom into the bivariate cluster plot:
par(mfrow = c(1,2))
clusplot(customer, fit$cluster, color = TRUE, shade = TRUE)
rect(-0.7, -1.7, 2.2, -1.2, border = "orange", lwd = 2)
clusplot(customer, fit$cluster, color = TRUE, xlim = c(-0.7,2.2), ylim = c(-1.7,-1.2))
##The clusplot function uses princomp and cmdscale to reduce the original feature dimension to the principal component. 
##Therefore, one can see how data is clustered in a single plot with these two components as the x-axis and y-axis.
par(mfrow = c(1, 1))
mds = cmdscale(dist(customer), k = 2)
plot(mds, col = fit$cluster)


# Comparing Clustering Sample
##After fitting data into clusters using different clustering methods, you may wish to measure the accuracy of the clustering. 
##In most cases, you can use either intracluster or intercluster metrics as measurements. 
##We will now introduce how to compare different clustering methods using cluster.stat from the fpc package.
library(fpc)
##We then need to use hierarchical clustering with the single method to cluster customer data and generate the object hc_single:
single_c =  hclust(dist(customer), method="single")
hc_single = cutree(single_c, k = 4)
##Use hierarchical clustering with the complete method to cluster customer data and generate the object hc_complete:
complete_c =  hclust(dist(customer), method="complete")
hc_complete =  cutree(complete_c, k = 4)
##We can then use k-means clustering to cluster customer data and generate the object km:
set.seed(22)
km = kmeans(customer, 4)
##Next, retrieve the cluster validation statistics of either clustering method:
cs = cluster.stats(dist(customer), km$cluster)
##Most often, we focus on using within.cluster.ss and avg.silwidth to validate the clustering method:
cs[c("within.cluster.ss","avg.silwidth")]
##Finally, we can generate the cluster statistics of each clustering method and list them in a table:
sapply(list(kmeans = km$cluster, hc_single = hc_single, hc_complete = hc_complete), 
       function(c)cluster.stats(dist(customer), c)[c("within.cluster.ss","avg.silwidth")])
##From the output, the within.cluster.ss measurement stands for the within clusters sum of squares, and avg.silwidth represents the average silhouette width.
##The within.cluster.ss measurement shows how closely related objects are in clusters
##silhouette is a measurement that considers how closely related objects are within the cluster and how clusters are separated from each other.
##The silhouette value usually ranges from 0 to 1; a value closer to 1 suggests the data is better clustered
km$withinss
km$betweenss
##Above shows that kmean function also contains withniss and betweens for us to see if the clustering is acceptable.

# Extracting Silhoutte Information from Clustering
##The silhouette coefficient combines the measurement of the intracluster and intercluster distance. 
##The output value typically ranges from 0 to 1; the closer to 1, the better the cluster is. 
##In this recipe, we will introduce how to compute silhouette information.
library(custer)
set.seed(2)
km = kmeans(customer, 4)
##We can then compute the silhouette information:
kms = silhouette(km$cluster, dist(customer))
summary(kms)
plot(kms)
##For those interested in how silhouettes are computed, please refer to the wikipedia entry for Silhouette Value: http://en.wikipedia.org/wiki/Silhouette_%28clustering%29


# Obtaining the Optimum Number of Clusters of K-means
##we can use the sum of squares to determine which k value is best for finding the optimum number of clusters for k-means. 
##In the following recipe, we will discuss how to find the optimum number of clusters for the k-means clustering method.
##---
##First, calculate the within sum of squares (withinss) of different numbers of clusters:
nk = 2:10
set.seed(22)
WSS = sapply(nk, function(k){kmeans(customer, centers = k)$tot.withinss})
WSS
##We can then use a line plot to plot the within sum of squares with a different number of k:
plot(nk, WSS, type ="l", xlab = "number of k", ylab = "within sum of squares")
##Next, we can calculate the average silhouette width (avg.silwidth) of different numbers of clusters:
SW = sapply(nk, function(k){cluster.stats(dist(customer), kmeans(customer, centers = k)$cluster)$avg.silwidth})
SW
##We can then use a line plot to plot the average silhouette width with a different number of k:
plot(nk, SW, type = "l", xlab = "number of clusers", ylab = "average silhouette width")
##Retrieve the maximum number of clusters:
nk[which.max(SW)]

# Clustering Data with the Density Based Method
##As an alternative to distance measurement, you can use a density-based measurement to cluster data. 
##This method finds an area with a higher density than the remaining area. One of the most famous methods is DBSCAN.
library(mlbench)
library(fpc)
##We can then use the mlbench library to draw a Cassini problem graph:
set.seed(2)
p = mlbench.cassini(500)
plot(p$x)
##Next, you can cluster data with regard to its density measurement:
ds = dbscan(dist(p$x), eps = 0.2, Minpts = 2, countmode = NULL, method = "dist")
ds
plot(ds, p$x)
##You can also use dbscan to predict which cluster the data point belongs to. 
##In this example, first make three inputs in the matrix p:
y = matrix(0,nrow = 3, ncol = 2)
y[1,] = c(0,0)
y[2,] = c(0,-1.5)
y[3,] = c(1,1)
y
##You can then predict which cluster the data belongs to:
predict(ds, p$x, y)
##Density-based clustering uses the idea of density reachability and density connectivity, which makes it very useful in discovering a cluster in nonlinear shapes.
##Density-based clustering takes two parameters into account: eps and MinPts.
##1. eps stands for the maximum radius of the neighborhood.
##2. MinPts denotes the minimum number of points within the eps neighborhood.
##For more introduction and theorm, please refer to the book.


# Clustering Data with a Model-based Method
##model-based clustering techniques assume varieties of data models and apply an EM algorithm to obtain the most likely model
##Model-based clustering assumes that the data is generated by an underlying probability distribution and tries to recover the distribution from the data.
library(mclust)
##We can then perform model-based clustering on the customer dataset:
mb = Mclust(customer)
plot(mb)
##Then, you can press the 1 key to obtain the BIC against a number of components:
##Then, you can press the 2 key to show the classification with regard to different combinations of features:
##Press the 3 key to show the classification uncertainty with regard to different combinations of features:
##Next, press the 4 key to plot the density estimation:
##Then, you can press the 0 key to plot density to exit the plotting menu.
##---
##The BIC plot shows the BIC value, and one can use this value to choose the number of clusters. 
##The classification plot shows how data is clustered in regard to different dimension combinations. 
##The uncertainty plot shows the uncertainty of classifications in regard to different dimension combinations. 
##The density plot shows the density estimation in contour.
##---
##Lastly, use the summary function to obtain the most likely model and number of clusters:
summary(mb)


# Visualizing a Dissimilarity Matrix
##A dissimilarity matrix can be used as a measurement for the quality of a cluster. 
##To visualize the matrix, we can use a heat map on a distance matrix. Within the plot, entries with low dissimilarity (or high similarity) are plotted darker, which is helpful to identify hidden structures in the data
library(seriation)
km = kmeans(customer, 4)
##You can then use dissplot to visualize the dissimilarity matrix in a heat map:
dissplot(dist(customer), labels = km$cluster, options = list(main = "Kmeans Clustering With k=4"))
##Next, apply dissplot on hierarchical clustering in the heat map:
complete_c =  hclust(dist(customer), method="complete")
hc_complete =  cutree(complete_c, k = 4)
dissplot(dist(customer), labels = hc_complete, options = list(main = "Hierarchical Clustering"))
##It shows that clusters similar to each other are plotted darker, and dissimilar combinations are plotted lighter.
##---
##Besides using dissplot to visualize the dissimilarity matrix, one can also visualize a distance matrix by using the dist and image functions.
image(as.matrix(dist(customer)))
##In order to plot both a dendrogram and heat map to show how data is clustered, you can use the heatmap function:
cd=dist(customer)
hc=hclust(cd)
cdt=dist(t(customer))
hcc=hclust(cdt)
heatmap(customer, Rowv=as.dendrogram(hc), Colv=as.dendrogram(hcc))

# Validating Clusters Externally
##Besides generating statistics to validate the quality of the generated clusters, we can use known data clusters as the ground truth to compare different clustering methods.
##In this recipe, we will continue to use handwriting digits as clustering inputs; we can find the figure at the author's GitHub page: https://github.com/ywchiu/ml_R_cookbook/tree/master/CH9
library(png)
##Read images from handwriting.png and transform the read data into a scatter plot:
img2 = readPNG("/Users/owner1/Desktop/R/Machine Learning of R Cookbook/Clustering Example Files/handwriting.png", T)
img3 = img2[, nrow(img2):1]
b = cbind(as.integer(which(img3 < -1) %% 28), which(img3 < -1) / 28)
plot(b, xlim=c(1,28), ylim=c(1,28))
##Perform a k-means clustering method on the handwriting digits:
set.seed(18)
fit = kmeans(b, 2)
plot(b, col=fit$cluster)
plot(b, col=fit$cluster,  xlim=c(1,28), ylim=c(1,28))
##Next, perform the dbscan clustering method on the handwriting digits:
library(fpc)
ds = dbscan(b, 2) > ds
plot(ds, b,  xlim = c(1,28), ylim=c(1,28))

