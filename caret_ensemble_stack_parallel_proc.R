rm(list=ls())  #--> for clean your environment
gc() #--> for launch the ''garbage collection''
# Load caret libraries
library(caret)
library(caretEnsemble)
library(readr) # CSV file I/O, e.g. the read_csv function
library(dplyr)
library(data.table)
library(ggplot2)
library(xgboost)
library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
library(MLmetrics)
library(tidyr)
library(stringr)
library(reshape)
library(mboost)
library(wrswoR)
#Semana, Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Demanda_uni_equil
setwd("D:/kaggle/test1/input/")
train=fread('../input/train.csv')
#test=fread('../input/test.csv')
#summary(train)
# Cut the train set to 8 and 9 weeks (Semana) for using only one week lags for target variable.
# If you have enough memory, you can set up condition Semana>3 on the next row for using lagged values of target variable for 5 weeks. 


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

trainData<-train[Semana == 8,]
#trainData <- trainData[1:500000,]
testData<-train[Semana ==9,]

xE = data.matrix(trainData[,c(1:5,7,9:14)]) 
yEf = trainData$target


## Rejection sampling
#s <- sample_int_crank(nrow(train), 2000000, train$target)
#training_sub2 <- training[s,]
#training_sub2=training_sub2[training_sub2$Semana<9,]
rm(train)
start_t<-Sys.time()
set.seed(100)

control <- trainControl(method="cv", number=2, verbose = TRUE, allowParallel=TRUE,     
                        savePredictions='final',index=createResample(yEf, 2))
algorithmList <- c('xgbLinear','gbm')
#algorithmList <- c('xgbLinear','glm', 'svmRadial')
set.seed(1234)

models <- caretList(x = xE,
                    y = yEf,
                    metric="RMSE",
                    trControl=control, 
                    verbose = TRUE,
                    methodList=algorithmList
)
results <- resamples(models)
summary(results)
modelCor(resamples(models))

####################################################

# stack using Logistics Regression
stackControl <- trainControl(method="repeatedcv", number=2, repeats=2, savePredictions="final",allowParallel=TRUE)#, classProbs=TRUE
set.seed(2341)
stack.glm <- caretStack(models, method="glm", metric="RMSE", trControl=stackControl)
print(stack.glm)

end_t<-Sys.time()
ensemble_time <- (end_t-start_t)
print(paste("Time:", print(ensemble_time)))
#rmsle xgbTree 0.28
#gbm_h2o
pred <- predict(stack.glm, newdata=testData[,c(1:5,7,9:14)])
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


pred <- predict(stack.glm, newdata=test[,c(1,2,4:13)])

#fix negatives

pred[pred<0] = 0.01

#create submission file
submission <- data.frame(ID=test$id, Demanda_uni_equil=round(pred))
write.csv(submission, "submission8.csv", row.names = F)
## When you are done:
stopCluster(cl)
