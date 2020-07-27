salary_train<-read.csv(file.choose()) # read train data
salary_test<-read.csv(file.choose()) # read test data
View(salary_train)
View(salary_test)
class(salary_train) # data.frame
class(salary_test) # data.frame
str(salary_train)
str(salary_test)
table(salary_train$Salary)
prop.table(table(salary_train$Salary)) # 75.10% , 24.90%
table(salary_test$Salary)
prop.table(table(salary_test$Salary)) # 75.43% , 24.57%

library(tm)

# Training on the model data 
library(e1071)
salary_classifier<-naiveBayes(salary_train,salary_train$Salary)
salary_classifier$levels

# Evaluating model performance
salary_test_pred<-predict(salary_classifier,salary_test)
salary_test_pred[1:10]

table_11<-table(salary_test_pred,salary_test$Salary)
table_11  # 247 + 202 errors
table(salary_test$Salary)

library(gmodels)
CrossTable(salary_test_pred,salary_test$Salary)

salary_classifier2<-naiveBayes(salary_train,salary_train$Salary,laplace = 11)
salary_test_pred2<-predict(salary_classifier2,salary_test)

table_12<-table(salary_test_pred2,salary_test$Salary)
table_12   # 257  + 156 errors
CrossTable(salary_test_pred2,salary_test$Salary)

salary_classifier3<-naiveBayes(salary_train,salary_train$Salary,laplace = 5)
salary_test_pred3<-predict(salary_classifier3,salary_test)

table_13<-table(salary_test_pred3,salary_test$Salary)
table_13   # 233  + 130 errors
CrossTable(salary_test_pred3,salary_test$Salary)

salary_classifier4<-naiveBayes(salary_train,salary_train$Salary,laplace = 1)
salary_test_pred4<-predict(salary_classifier4,salary_test)

table_14<-table(salary_test_pred4,salary_test$Salary)
table_14   # 174  + 97 errors
CrossTable(salary_test_pred4,salary_test$Salary)

# Accuracy
accuracy11<-(sum(diag(table_11))/sum(table_11))
accuracy11  # 97.01%
accuracy12<-(sum(diag(table_12))/sum(table_12))
accuracy12   # 97.25%
accuracy13<-(sum(diag(table_13))/sum(table_13))
accuracy13   # 97.58%
accuracy14<-(sum(diag(table_14))/sum(table_14))
accuracy14   # 98.20%

## Based on above all models laplace=1 model is more suitable 
# considering the less number of errors & high accuracy.
