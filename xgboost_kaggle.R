rm(list=ls())  #--> for clean your environment
gc() #--> for launch the garbage collection
library(readr) # CSV file I/O, e.g. the read_csv function
library(MLmetrics)
library(data.table)
library(xgboost)
print(paste("Load Data",Sys.time()))
setwd("D:/kaggle/test1/input/")
## load the training file, using just the fields available for test
train<-fread("../input/train.csv"
             ,select = c("Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Demanda_uni_equil","Agencia_ID"))
train[,target:=log1p(Demanda_uni_equil)]

Sys.time()
productInfo<-train[Semana < 8,.(nProduct=.N,productMeanLog=mean(target)),Producto_ID]
Sys.time()
clientInfo<-train[Semana < 8,.(nClient=.N,clientMeanLog=mean(target)),Cliente_ID]
Sys.time()
productClientInfo<-train[Semana < 8,.(nProductClient=.N,productClientMeanLog=mean(target)),.(Producto_ID,Cliente_ID)]
Sys.time()
## remove all weeks used for creating the averages, and create a modeling set with the remaining data
train<-train[Semana >= 8,]
## now add the features we have created from weeks 3-7 to weeks 8 and 9
train<-merge(train[Semana >= 8],productInfo,by="Producto_ID",all.x=TRUE)
Sys.time()
train<-merge(train,clientInfo,by="Cliente_ID",all.x=TRUE)
Sys.time()
train<-merge(train,productClientInfo,by=c("Cliente_ID","Producto_ID"),all.x=TRUE)
Sys.time()
train[is.na(nProduct),nProduct:=0]
train[is.na(productMeanLog),productMeanLog:=0]
train[is.na(nClient),nClient:=0]
train[is.na(clientMeanLog),clientMeanLog:=0]
train[is.na(nProductClient),nProductClient:=0]
train[is.na(productClientMeanLog),productClientMeanLog:=0]

#################
## Set up Cluster (H2O is a Java ML Platform, with R/Python/Web/Java/Spark/Hadoop APIs)
#################
#library(h2o)
#h2o.init(nthreads=-1,max_mem_size = '5G')
#dev<-as.h2o(train[Semana<=8,],destination_frame = "dev.hex")
#val<-as.h2o(train[Semana==9,],destination_frame = "val.hex")

#training <- train[Semana==8,]
#testing <- train[Semana==9,]

trainData<-train[Semana == 8,]
#trainData <- trainData[1:500000,]
testData<-train[Semana ==9,]

#xE = data.matrix(trainData[,c(1:5,7,9:14)]) 
#yEf = trainData$target
rm(train)
dtrain <- xgb.DMatrix(data = data.matrix(trainData[,c(1:5,7,9:14)]), label=data.matrix(trainData$target))
dtest <- xgb.DMatrix(data = data.matrix(testData[,c(1:5,7,9:14)]), label=data.matrix(testData$target))
rm(trainData)
watchlist <- list(train=dtrain, test=dtest)

clf <- xgb.train(params=list(  objective="reg:linear", 
                               #booster = "gblinear",
                               booster = "gbtree",
                               eta=0.1, 
                               max_depth=30,
                               subsample=0.85,
                               colsample_bytree=0.7
) 
,
data = dtrain, 
nrounds = 150, 
verbose = 1,
print_every_n=5,
early_stopping_rounds    = 10,
watchlist           = watchlist,
maximize            = FALSE,
eval_metric='rmse'
)

# Make prediction for the 10th week
#data_test1=data_test[Semana==10,]

#testing$Ruta_SAK <- mean(testing$Ruta_SAK)
#testing$Cliente_ID <- mean(testing$Cliente_ID)
#testing$Producto_ID <- mean(testing$Producto_ID)

#pred<-predict(clf,xgb.DMatrix(data.matrix(testing),missing=NA))
pred<-predict(clf,xgb.DMatrix(data.matrix(testData[,c(1:5,7,9:14)]),missing=NA))
#pred <- predict(clf, newdata=testData[,c(1:5,7,9:14)])
pred[pred<0] = 0.01

#check RMSLE on test data
rmtest <- RMSLE(pred, testData$target)
print(rmtest)
############################################################################
test<-fread("../input/test.csv")
test[1:2,] ## take a look at a few rows of the test data
## merge in the offset column, just as with val and final
## now add the features we have created from weeks 3-7 to weeks 8 and 9
test<-merge(test,productInfo,by="Producto_ID",all.x=TRUE)
Sys.time()
test<-merge(test,clientInfo,by="Cliente_ID",all.x=TRUE)
Sys.time()
test<-merge(test,productClientInfo,by=c("Cliente_ID","Producto_ID"),all.x=TRUE)[order(id)]
Sys.time()
test[is.na(nProduct),nProduct:=0]
test[is.na(productMeanLog),productMeanLog:=0]
test[is.na(nClient),nClient:=0]
test[is.na(clientMeanLog),clientMeanLog:=0]
test[is.na(nProductClient),nProductClient:=0]
test[is.na(productClientMeanLog),productClientMeanLog:=0]


#pred <- predict(stack.glm, newdata=test[,c(1,2,4:13)])
prediction <- predict(clf, xgb.DMatrix(data.matrix(test[,c(1,2,4,6,7,5,8,9:13)]),missing=NA))

#fix negatives

prediction[prediction<0] = 0.01

#create submission file
submission <- data.frame(ID=test$id, Demanda_uni_equil=round(prediction))
write.csv(submission, "submission10.csv", row.names = F)

