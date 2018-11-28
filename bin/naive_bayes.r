setwd(".")
options(stringsAsFactors = FALSE)
# library("clusterSim")
# library("PRROC")
library("e1071")

source("./confusion_matrix_rates.r")

threshold <- 0.5

cat("threshold = ", threshold, "\n", sep="")

fileName <- "../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv"
prc_data_norm <- read.csv(file=fileName,head=TRUE,sep=",",stringsAsFactors=FALSE)

prc_data_norm <- prc_data_norm[sample(nrow(prc_data_norm)),] # shuffle the rows

target_index <- dim(prc_data_norm)[2]

training_set_perce = 80
cat("training_set_perce = ", training_set_perce, "%\n", sep="")

# the training set is the first 60% of the whole dataset
training_set_first_index <- 1 # NEW
training_set_last_index <- round(dim(prc_data_norm)[1]*training_set_perce/100) # NEW

# the test set is the last 40% of the whole dataset
test_set_first_index <- training_set_last_index+1 # NEW
test_set_last_index <- dim(prc_data_norm)[1] # NEW

cat("[Creating the subsets for the values]\n")
prc_data_train <- prc_data_norm[training_set_first_index:training_set_last_index, 1:(target_index)] # NEW
prc_data_test <- prc_data_norm[test_set_first_index:test_set_last_index, 1:(target_index)] # NEW

cat("[Creating the subsets for the labels \"1\"-\"0\"]\n")
prc_data_train_labels <- prc_data_norm[training_set_first_index:training_set_last_index, target_index] # NEW
prc_data_test_labels <- prc_data_norm[test_set_first_index:test_set_last_index, target_index]   # NEW


print("dim(prc_data_train)")
print(dim(prc_data_train))

print("dim(prc_data_test)")
print(dim(prc_data_test))


library(class)
library(gmodels)

naive_bayes_model <-  naiveBayes(as.factor(Metastasis) ~ . , data=prc_data_train)

prc_data_test_PRED <- predict((naive_bayes_model), prc_data_test)
prc_data_test_PRED_binary <- as.numeric(prc_data_test_PRED)-1

prc_data_test_PRED_binary[prc_data_test_PRED_binary>=threshold]=1
prc_data_test_PRED_binary[prc_data_test_PRED_binary<threshold]=0

# print("predictions:")
# print(prc_data_test_PRED_binary)
# 
# 
# print("labels:")
# print(prc_data_test$Metastasis)


confusion_matrix_rates(prc_data_test_labels, prc_data_test_PRED_binary, "@@@ Test set @@@")



