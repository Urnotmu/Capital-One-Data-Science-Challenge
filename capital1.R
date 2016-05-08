#File import - codetest_train

library(psych)
describe(codetest_train)

#target is continuous ..not a classification problem
#f_61, f_!21, f_215, f_237 are categorical variables
#All the nummerical data provided to me are normalized and given, target has a very high variance
#Imputation is required to run a model, since around 200 obs from each attribute are missing - regression not possible

#Since, data in standardized,dimensional reduction may not work, hence, imputation might be important for better accuracy
#Step - 1 : Imputation using MICE

library(mice)


Micedata<- mice(codetest_train[,-1], m = 3, maxit = 2, method = "pmm", seed = 1234)

imputeddata1<- complete(Micedata,1)
imputeddata2<- complete(Micedata,2)
imputeddata3<- complete(Micedata,3)


#Have removed the target variable to eliminate the imputation bias

#Categorical variables are not imputed, will run a regression to understand the significance of the 4 variables
# R will convert the factor levels automatically
#Using cbind the add the target column to the imputed dataset for sample regression

target<- codetest_train[,1]
Regdata1<-cbind(target, imputeddata1)
Regdata2<-cbind(target, imputeddata2)
Regdata3<-cbind(target, imputeddata3)

Reg1<- lm(target~., Regdata1)
summary(Reg1) #Adj Rsq = 58.88%

Reg2<- lm(target~., Regdata2)
summary(Reg2) #Adjusted Rsq = 58.63%

Reg3<- lm(target~., Regdata3)
summary(Reg3)   #Adj Rsq = 58.64%

#Using imputedata1

ImputedData<- imputeddata1
Regdata<- cbind(target, ImputedData)

#Too many variables, hence adjusted Rsq - 58.88%. should be ok for a real time std data
#f_61 & f_237 seem slightly significant..

#PCA and Stepwise forward and backward regression have not yielded any result, since the data in normalized, each variable s explanatory power is reduced
#Removing my variables will reduce my accuracy since all the variables are contribution variance at a lower level, will retail the variables

#Partioning the data into Training(4000) and validation set(1000) to check for model accuracy
#Cross Validation not necessary


  sampleInstances<-sample(1:nrow(Regdata),size = 0.8*nrow(Regdata))
  trainData<-Regdata[sampleInstances,]
  testData<-Regdata[-sampleInstances,]
  
#Model Comparison
  
#1. Linear Regression
  
  modelfit<-lm(target~.,trainData)
  summary(modelfit) #Adj Rsq = 59.25
  
  prediction<-predict(modelfit, testData[,-1])
  mse<- mean((testData[,1] - prediction)^2) #Mean Squared Error
  mse # [1] 10.37713    [1] 12.11896



#2. Support Vector Machine Regression

library(kernlab)
modelfit<- ksvm(target~.,trainData)
summary(modelfit) 

prediction<-predict(modelfit, testData[,-1])
mse<- mean((testData[,1] - prediction)^2) #Mean Squared Error
mse # [1] 11.47762


#3. Neural Nets Regression

library(nnet)

modelfit<- nnet(target~., trainData, size = 3, maxit = 5)
summary(modelfit)
prediction<-predict(modelfit, testData[,-1])
mse<- mean((testData[,1] - prediction)^2) #Mean Squared Error
mse # [1] 26.4432

#4. Random Forest Regression

library(randomForest)
modelfit<- randomForest(target~., trainData)
summary(modelfit)

prediction<-predict(modelfit, testData[,-1])
mse<- mean((testData[,1] - prediction)^2) #Mean Squared Error
mse # [1] 11.96596

#5. Gradient Boosted Regression

library(gbm)

modelfit<-gbm(target~., traindata, distribution = "gaussian")
summary(modelfit)

mse<- mean((testData[,1] - prediction)^2) #Mean Squared Error
mse # [1] 11.96596

#SVM regression has consistently produced lower MSE result. Hence will use SVM Regression(OLP varies with each different sample)

#Using SVM Regression

# Performing MI for test data

library(mice)
Micedata<- mice(codetest_test, m = 3, maxit = 2, method = "pmm", seed = 1234)

# Taking three imputed Outputs

codetest1<- complete(Micedata,1)
codetest2<- complete(Micedata,2)
codetest3<- complete(Micedata,3)

#Modelcomparison by SVM Regression


library(kernlab)
modelfit<- ksvm(target~.,trainData)
summary(modelfit) 

prediction1<-predict(modelfit, codetest1)
prediction2<-predict(modelfit, codetest2)
prediction3<-predict(modelfit, codetest3)

prediction1<- as.matrix(prediction1)
prediction2<- as.matrix(prediction2)
prediction3<- as.matrix(prediction3)


#Averaging the 3 outputs will lower the probability of obtaining the highest MSE among the 3 testdata sample

predictionmain<- (prediction1+prediction2+prediction3)/3
predictionmain<- data.frame(predictionmain)

write.table(predictionmain,file = "predictionmain.txt") #Output file









