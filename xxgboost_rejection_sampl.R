rm(list=ls())  #--> for clean your environment
gc() #--> for launch the ''garbage collection''
library(readr) # CSV file I/O, e.g. the read_csv function
library(MLmetrics)
library(data.table)
library(xgboost)
library(wrswoR)
#Semana, Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Demanda_uni_equil
setwd("D:/kaggle/test1/input/")
train=fread('../input/train.csv')
# Cut the train set to 8 and 9 weeks (Semana) for using only one week lags for target variable.
# If you have enough memory, you can set up condition Semana>3 on the next row for using lagged values of target variable for 5 weeks. 
Y = (log1p(train[,11]))
train$Y <- Y
rm(Y)
training=train[Semana<9,]
training <- as.data.frame(training[,c(1,2,3,4,5,6,12)])

testing=train[Semana==9,]
training_sub <- as.data.frame(training_sub[,c(1,2,3,4,5,6,12)])
testing <- as.data.frame(testing[,c(1,2,3,4,5,6,12)])

## Rejection sampling
s <- sample_int_crank(nrow(training), 10000000, training$Y)
training_sub2 <- training[s,]
#training_sub2=training_sub2[training_sub2$Semana<9,]
rm(train)
start_t<-Sys.time()
set.seed(100)


dtrain <- xgb.DMatrix(data = data.matrix(training_sub2[1:6]), label=data.matrix(training_sub2$Y))
dtest <- xgb.DMatrix(data = data.matrix(testing[1:6]), label=data.matrix(testing$Y))

watchlist <- list(train=dtrain, test=dtest)

clf <- xgb.train(params=list(  objective="reg:linear", 
                               #booster = "gblinear",
                               booster = "gbtree",
                               eta=0.1, 
                               max_depth=10 
                               #subsample=0.85,
                               #colsample_bytree=0.7
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


pred<-predict(clf,xgb.DMatrix(data.matrix(testing[,c(1:6)]),missing=NA))

ensemble_time <- (end_t-start_t)
print(paste("Time:", print(ensemble_time)))


#fix negatives
pred[pred<0] = 0.01

rm <- RMSLE(pred, testing$Y)

# read in the test data
test=fread('../input/test.csv')

test3 <- test[,c(2:7)]
head(test3)

prediction <- predict(clf, xgb.DMatrix(data.matrix(test3),missing=NA))
str(prediction)

#fix negatives
prediction[prediction<0] = 0.01

#create submission file

submission <- data.frame(ID=test$id, Demanda_uni_equil=round(prediction))

write.csv(submission, "D:/kaggle/test1/xgboost_rejectionsmpl2.csv", row.names = F)

