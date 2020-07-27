
sms_data<-read.csv(file.choose(),stringsAsFactors = F) # Read sms_data.csv
class(sms_data) # data frame
str(sms_data) # both variables found datatype as chr
sms_data$type<-factor(sms_data$type) # convert 'Type' fom chr to factor
str(sms_data) # found datatype of 'Type' as factor & 'Text' remains same as chr
table(sms_data$type) # ham= 4812 , spam=747

library(tm)

# Prepare corpous for the text data
sms_corpous<-Corpus(VectorSource(sms_data$text))
sms_corpous$content[1:20]

# Cleaning the data(removing the unwanted symbols)
corpus_clean<-tm_map(sms_corpous,tolower)
corpus_clean<-tm_map(corpus_clean,removeNumbers)
corpus_clean<-tm_map(corpus_clean,removeWords,stopwords())
corpus_clean<-tm_map(corpus_clean,removePunctuation)
removeNumPunct<-function(x) gsub("[^[:alpha:][:space:]]*","",x)
corpus_clean<-tm_map(corpus_clean,content_transformer(removeNumPunct))
corpus_clean<-tm_map(corpus_clean,stripWhitespace)
class(corpus_clean)

corpus_clean$content[1:10]

# Use DTM (Document term matrix)
sms_dtm<-DocumentTermMatrix(corpus_clean) 
class(sms_dtm)

# Raw sms data splitted in Train & Test (75/25 ratio)
sms_raw_train<-sms_data[1:4169, ] 
sms_raw_test<-sms_data[4170:5559, ]

# dtm data splitted in Train & Test (75/25 ratio)
sms_dtm_train<-sms_dtm[1:4169, ]
sms_dtm_test<-sms_dtm[4170:5559, ]

# clean corpus data splitted in Train & Test (75/25 ratio)
sms_corpus_train<-corpus_clean[1:4169]
sms_corpus_test<-corpus_clean[4170:5559]

# check the proportion of spam is similar
prop.table(table(sms_data$type)) # ham=0.87 , spam=0.13
prop.table(table(sms_raw_train$type)) # ham=0.87 , spam=0.13
prop.table(table(sms_raw_test$type)) # ham=0.87 , spam=0.13

# Indicator feature for frequent words
sms_dict<-findFreqTerms(sms_dtm_train,3)
list(sms_dict[1:100])

sms_train<-DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test<-DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

#convert counts to factor
convert_counts<- function(x) {
  x<-ifelse(x>0,1,0)
  x<-factor(x,levels = c(0,1), labels = c("No","Yes"))
}

# apply () convert_counts() to columns of train & test data
sms_train<-apply(sms_train,MARGIN = 2,convert_counts)
sms_test<-apply(sms_test,MARGIN = 2,convert_counts)
View(sms_train)
View(sms_test)

########################## BUILD NAIVE BAYES MODEL ############################

# Training on the model data 
library(e1071)
sms_classifier<-naiveBayes(sms_train,sms_raw_train$type)
sms_classifier$levels # ham , spam

# Evaluating model performance
sms_test_pred<-predict(sms_classifier,sms_test)
sms_test_pred[1:25]

table1<-table(sms_test_pred,sms_raw_test$type)
table1 # FN=4 , FP=28
table(sms_raw_test$type)

# Accuracy
accuracy1<-(sum(diag(table1))/sum(table1))
accuracy1 # 0.9769

library(gmodels)
CrossTable(sms_test_pred,sms_raw_test$type)

################### BUILD ANOTHER MODEL BY USING LAPLACE = 11 ###############

# Build another model by using laplace to check the improvement in accuracy/error
sms_classifier2<-naiveBayes(sms_train,sms_raw_train$type, laplace = 11)
sms_test_pred2<-predict(sms_classifier2,sms_test)
table2<-table(sms_test_pred2,sms_raw_test$type)
table2 # FN=0 , FP=177

# Accuracy
accuracy2<-(sum(diag(table2))/sum(table2))
accuracy2 # 0.8726

CrossTable(sms_test_pred2,sms_raw_test$type)

################### BUILD ANOTHER MODEL BY USING LAPLACE = 4 ###############

# Build another model by using laplace to check the improvement in accuracy/error
sms_classifier3<-naiveBayes(sms_train,sms_raw_train$type, laplace = 4)
sms_test_pred3<-predict(sms_classifier3,sms_test)
table3<-table(sms_test_pred3,sms_raw_test$type)
table3 # FN=5 , FP=68

# Accuracy
accuracy3<-(sum(diag(table3))/sum(table3))
accuracy3 # 0.9474

CrossTable(sms_test_pred3,sms_raw_test$type)

################### BUILD ANOTHER MODEL BY USING LAPLACE = 2 ###############

# Build another model by using laplace to check the improvement in accuracy/error
sms_classifier4<-naiveBayes(sms_train,sms_raw_train$type, laplace = 2)
sms_test_pred4<-predict(sms_classifier4,sms_test)
table4<-table(sms_test_pred4,sms_raw_test$type)
table4 # FN=3 , FP=43

# Accuracy
accuracy4<-(sum(diag(table4))/sum(table4))
accuracy4 # 0.9669

CrossTable(sms_test_pred4,sms_raw_test$type)

############################# END ############################