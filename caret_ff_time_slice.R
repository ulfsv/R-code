# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 
rm(list=ls())  #--> for clean your environment
gc() #--> for launch the ''garbage collection''
#library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret)
library(doParallel)
library(ff)
library(ffbase)
library(MLmetrics)
#library(devtools)
#install_github("edwindj/ffbase", subdir="pkg")
# In##ut data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Set Working Directory to where big data is
setwd("D:/kaggle/data/")

# Check temporary directory ff will write to (avoid placing on a drive with SSD)
getOption("fftempdir")

# Set new temporary directory
options(fftempdir = "D:/kaggle/data/temp")

# Load in the big data
sampleData <- read.csv("train.csv", header = TRUE, nrows = 5, sep=",")
classes <- sapply(sampleData, class)

train_data = read.csv.ffdf(file="train.csv", # File Name
                           sep=",",         # Tab separator is used
                           header=T,     # No variable names are included in the file
                           fill = TRUE,      # Missing values are represented by NA
                           colClasses = classes
                        , nrows=10000
                           # Specify the import type of the data
)
class(train_data)
dim(train_data)


#Step 1: Creating the timeSlices for the index of the data:

timeSlices <- createTimeSlices(1:nrow(train_data), 
                               initialWindow = 1000, horizon = 1000, fixedWindow = F)

#This creates a list of training and testing timeSlices.
str(timeSlices,max.level = 1)

#rm(client_table, product_table,town_table,test_data)
## List of 2
## $ train:List of 431
##   .. [list output truncated]
## $ test :List of 431
##   .. [list output truncated]
#For ease of understanding, I am saving them in separate variable:

trainSlices <- timeSlices[[1]]
testSlices <- timeSlices[[2]]

#Step 2: Training on the first of the trainSlices:
#Y <- (train_data$Demanda_uni_equil)
#train_data <- (train_data[,c(1:6)])

# Get predictor variable names (only 1 categorical is included)
data_variables = colnames(train_data)[c((1:6))]

# Create model formula statement
model_formula = as.formula(paste0("Demanda_uni_equil ~", paste0(data_variables, collapse="+")))


plsFitTime <- train(model_formula,
                    data = train_data[trainSlices[[1]],],
                    method = "rpart",
                    allowParallel=TRUE,
                    preProc = c("center", "scale"))
#Step 3: Testing on the first of the trainSlices:

pred <- predict(plsFitTime,economics[testSlices[[1]],])

#Step 4: Plotting:

true <- train_data$Y[testSlices[[1]]]

plot(true, col = "red", ylab = "true (red) , pred (blue)", ylim = range(c(pred,true)))
points(pred, col = "blue") 

#You can then do this for all the slices:

for(i in 1:length(trainSlices)){
  plsFitTime <- train(unemploy ~ pce + pop + psavert,
                      data = economics[trainSlices[[i]],],
                      method = "glm",
                      allowParallel=TRUE,
                      preProc = c("center", "scale"))
  
  pred <- predict(plsFitTime,economics[testSlices[[i]],])
  
  
  true <- economics$unemploy[testSlices[[i]]]
  
  plot(true, col = "red", ylab = "true (red) , pred (blue)", 
       main = i, ylim = range(c(pred,true)))
  points(pred, col = "blue") 
}

stopCluster(cl)

#check RMSLE on test data
rmtest <- RMSLE(prediction, testDatatarget$x)
print(rmtest)




#create submission file
submission <- data.frame(ID=testLargeData$id, Demanda_uni_equil=prediction)
write.csv(submission, "submission.csv", row.names = F)

# Any results you write to the current directory are saved as output.
