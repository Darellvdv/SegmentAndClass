options(warn=-1)
suppressMessages(if (!require("pacman")) install.packages("pacman"))
pacman::p_load(stringr, rgdal, rgeos, raster, maptools, zoo, igraph, leaflet)

args <- commandArgs(trailingOnly = TRUE)

inputshp <- args[1]
inputcsv <- args[2]
layername <- args[3]

# Load Shapefile
shape <- readOGR(dsn= inputshp, layer = layername)
shape_df <- as.data.frame(shape)
#shape$label2 <- 1:nrow(shape_df)

# Load numpy array
attributes <- read.table(inputcsv, header = FALSE, sep = ",")

# Create rounding function
round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  
  df[,nums] <- round(df[,nums], digits = digits)
  
  (df)
}

# Round df
rounddf <- round_df(attributes, digits=3)
colnames(rounddf) <- c('NewIndex', 'Oldindex', 
                       'NDVI', 'Greenness', 
                       'MeanR', 'MeanG', 'MeanB', 'MeanNIR', 
                       'VarR', 'VarG', 'VarB', 'VarNIR',
                       'R/G', 'G/R', 'B/NIR', 'G/NIR',
                       'Perimeter', 'Area', 'Extent', 'Eccentricity', 'Roughness', 'VissibleBirght', 
                       'NDVIdiff', 'Greennessdiff', 'Texturediff', 'Brightnessdiff')

# Merge attributes
m <- merge(shape, rounddf, by.x = "label", by.y = "Oldindex")

inputshpsplit <- unlist(strsplit(inputshp, '[.]'))
writeOGR(m, paste(inputshpsplit,'_allvall.shp'), paste(inputshpsplit,'_allvall'), driver = 'ESRI Shapefile')

print("Shapefile merged, terminating R script!")


#---------------#
# Train dataset #
#......

# THen:

#library(rpart)
#library(rpart.plot)
#library(caret)

#data <- read.csv('D:/pinapple/firstoutputtest8-2/training_select_roadgap.csv')
#data <- na.omit(data)

#train <- data[sample(nrow(data), 300), ]
#test <- subset(data, !(data$label %in% train$label))


#model1 <- rpart(class~NDVI+GRVI+meanB0+meanB1+meanB2+meanB3+varB0+varB1+varB2+varB3+Perimeter+Area+Extent+Ecc+NDVIdiff+Greennesdiff, 
#                method='class', data = train, control = rpart.control(minsplit = 20, minbucket = round(20/3), cp = 0.01, 
#                                                                      maxcompete = 18, maxsurrogate = 5, usesurrogate = 2, xval = 10,
#                                                                      surrogatestyle = 0, maxdepth = 30))

#model1pred <- predict(model1, data, methods='')


#pred <- as.numeric(colnames(model1pred)[apply(model1pred,1,which.max)])
#cor <- cor(data[,19],pred)

#rpart.plot(model1)

#conmatr <- confusionMatrix(data[,19],pred)

#summary(model1)