rm(list=ls())  #--> for clean your environment
gc() #--> for launch the garbage collection
library(readr) # CSV file I/O, e.g. the read_csv function
library(MLmetrics)
library(data.table)
library(mxnet)
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
trainData<-train[Semana == 8,]
#trainData <- trainData[1:500000,]
testData<-train[Semana ==9,]

#xE = data.matrix(trainData[,c(1:5,7,9:14)]) 
#yEf = trainData$target
rm(train)
#dtrain <- xgb.DMatrix(data = data.matrix(trainData[,c(1:5,7,9:14)]), label=data.matrix(trainData$target))
#dtest <- xgb.DMatrix(data = data.matrix(testData[,c(1:5,7,9:14)]), label=data.matrix(testData$target))

dtrain <- data.matrix(trainData[,c(1:5,7,9:14)])
dtrainy <- (trainData$target)
 dtest <-data.matrix(testData[,c(1:5,7,9:14)])  
 dtesty <- (testData$target)
#rm(trainData)

mx.callback.plot.train.metric <- function(period, logger = NULL){
  function(iteration, nbatch, env, verbose=TRUE){
    if (nbatch %% period == 0 && !is.null(env$metric)){
      N = env$end.round
      result <- env$metric$get(env$train.metric)
      #plot(c(0.5, 1) ~ c(0, N), col=NA, ylab=paste0("Train ", result$name), xlab="")
      plot(c(1.0, 0.0) ~ c(0, N), col=NA, ylab=paste0("Train ", result$name), xlab="")
      logger$train <- c(logger$train, result$value)
      lines(logger$train, lwd=3, col="red")
    }
    return(TRUE)
  }
}
# log-loss
mx.metric.rmse <- mx.metric.custom("rmse", function(label, pred){
  return(Metrics::rmse(label, pred))
})
logger <- mx.metric.logger$new()
#tic <- proc.time()
mx.set.seed(2018)
data <- mx.symbol.Variable("data")

Dropout1 <- mx.symbol.Dropout(data, p=0.02, name='Dropout1') 
fc3 <- mx.symbol.FullyConnected(Dropout1, name="fc3", num_hidden=20)
bn_1 <- mx.symbol.BatchNorm(fc3, name='bn_1') 
act3 <- mx.symbol.Activation(bn_1, name="relu3", act_type="relu")

Dropout2 <- mx.symbol.Dropout(act3, p=0.01, name='Dropout2') 
fc4 <- mx.symbol.FullyConnected(Dropout2, name="fc4", num_hidden=10)
bn_2 <- mx.symbol.BatchNorm(fc4, name='bn_2') 
act4 <- mx.symbol.Activation(bn_2, name="relu4", act_type="relu")

Dropout3 <- mx.symbol.Dropout(act4, p=0.02, name='Dropout3')
fc5 <- mx.symbol.FullyConnected(Dropout3, name="fc5", num_hidden=10)
bn_3 <- mx.symbol.BatchNorm(fc5, name='bn_3') 
act5 <- mx.symbol.Activation(bn_3, name="relu5", act_type="relu")

Dropout4 <- mx.symbol.Dropout(act5, p=0.02, name='Dropout4') 

fc6 <- mx.symbol.FullyConnected(Dropout4, name="fc6", num_hidden=1)
#linear_reg_output <- mx.symbol.SoftmaxOutput(fc6, name="sm")
linear_reg_output <- mx.symbol.LinearRegressionOutput(fc6, name="lin")
#model <- paste(fname2d2, "BTChart.Rdata", sep="")
model <- mx.model.FeedForward.create(symbol = linear_reg_output, dtrain, dtrainy,
                                     #begin.round = 2,
                                     #optimizer = "adagrad",
                                     #optimizer         = "sgd",
                                     #optimizer         = "rmsprop",
                                     optimizer         = "adam",
                                     ctx = mx.cpu(), 
                                     num.round = 2,
                                     array.batch.size = 100,
                                     #learning.rate = 0.07,
                                     #learning.rate = exp(-1),
                                     #learning.rate = 1,
                                     #0.001
                                     # 0.07
                                     #momentum = 0.94,
                                     #wd = 0.00001,
                                     verbose = FALSE,
                                     #optimizer = "rmsprop",
                                     eval.metric =  mx.metric.rmse,
                                     initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1),
                                     batch.end.callback = mx.callback.log.train.metric(100),
                                     epoch.end.callback = mx.callback.plot.train.metric(100, logger))
#batch_end_callback=mx.callback.Speedometer(1, 10))
logger_df <- data.frame(Interation=1:length(logger$train), Accuracy = logger$train, Train_error = (1-logger$train))
#print(proc.time() - tic)
#setwd("D:/kaggle/test1/input/") #spara
mx.model.save(
  model = model,
  prefix = "test2",
  iteration = 2)
model <- mx.model.load(prefix = "test", iteration = 2)
test <- testData[,c(1:5,7,9:14)]
test <- test[1:2000000,]
test2 <- data.matrix(test)
preds_rmsprop <- predict(model, test2, ctx=mx.gpu())
dim(preds_rmsprop)

# Make prediction for the 10th week
#data_test1=data_test[Semana==10,]

#testing$Ruta_SAK <- mean(testing$Ruta_SAK)
#testing$Cliente_ID <- mean(testing$Cliente_ID)
#testing$Producto_ID <- mean(testing$Producto_ID)

#pred<-predict(clf,xgb.DMatrix(data.matrix(testing),missing=NA))
#pred<-predict(clf,xgb.DMatrix(data.matrix(testData[,c(1:5,7,9:14)]),missing=NA))
#pred <- predict(clf, newdata=testData[,c(1:5,7,9:14)])
pred[pred<0] = 0.01

#check RMSLE on test data
dtesty2 <- dtesty[1:2000000]
rmtest <- RMSLE(preds_rmsprop, dtesty2)
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
#prediction <- predict(clf, xgb.DMatrix(data.matrix(test[,c(1,2,4,6,7,5,8,9:13)]),missing=NA))
prediction <- predict(model, data.matrix(test[,c(1,2,4,6,7,5,8,9:13)]), ctx=mx.cpu())
#fix negatives

prediction[prediction<0] = 0.01
pred <- as.data.frame(prediction)
#create submission file

submission <- data.frame(ID=test$id, Demanda_uni_equil=round(pred$V1))
write.csv(submission, "submission12.csv", row.names = F)

