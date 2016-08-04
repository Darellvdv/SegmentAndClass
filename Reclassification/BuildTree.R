library(rpart)
library(rpart.plot)
library(caret)

data <- read.csv('D:/pinapple/Trial_Dole/train/allclasses.csv')
data <- na.omit(data)

train <- data[sample(nrow(data), 250), ]
test <- subset(data, !(data$label %in% train$label))


model1 <- rpart(class~NDVI+Grennss+MeanR+MeanG+MeanB+MeanNIR+VarR+VarG+VarG+VarNIR+RG+GR+BNIR+GNIR+Perimtr+Area+Extent+Eccntrc+Roghnss+VssblBr+NDVIdff+Grnnssd+Txtrdff+Brghtns, 
                method='class', data = train, control = rpart.control(minsplit = 5, cp = 0.01, 
                                                                     maxcompete = 250, maxsurrogate = 30, usesurrogate = 0, xval = 10,
                                                                     surrogatestyle = 0, maxdepth = 4))
rpart.plot(model1)
summary(model1)

model1pred <- predict(model1, test, methods='')


pred <- as.numeric(colnames(model1pred)[apply(model1pred,1,which.max)])
cor <- cor(test[,1],pred)

#rpart.plot(model1)
conmatr <- table(factor(pred, levels=min(test[,1]):max(test[,1])),factor(test[,1], levels=min(test[,1]):max(test[,1])))
conmatr <- confusionMatrix(test[,1],pred)

library(raster)
ras <- raster("D:/pinapple/Trial_Dole/NDVItest/output/test_noalpha.tif")
ndvi <- raster("D:/pinapple/Trial_Dole/test_ndvi.tif")

stac <- stack(ras,ndvi)
stac2 <- addLayer(ras, ndvi)


writeRaster(stac, 'D:/pinapple/Trial_Dole/NDVItest/rnirndvi.tif')

stac <- stack("D:/pinapple/Trial_Dole/NDVItest/output/test_noalpha.tif")

ras[[2]]
