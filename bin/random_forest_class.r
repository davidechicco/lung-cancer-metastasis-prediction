setwd(".")
options(stringsAsFactors = FALSE)
# library("clusterSim")
library("PRROC")
library("e1071")
library("randomForest");

source("./confusion_matrix_rates.r")
source("./utils.r")

# args = commandArgs(trailingOnly=TRUE)
# thisNtree <- as.integer(args[1])

thisNtree <- 5000

threshold <- 0.5

cancer_data_norm <- read.csv(file="../data/LungCancerDataset_AllRecords_NORM_reduced_features.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

cat("[Randomizing the rows]\n")
cancer_data_norm <- cancer_data_norm[sample(nrow(cancer_data_norm)),] # shuffle the rows

dataset_dim_retriever(cancer_data_norm)
imbalance_retriever(cancer_data_norm$Metastasis)

target_index <- dim(cancer_data_norm)[2]

target_label <- colnames(cancer_data_norm[target_index])


training_set_perc=60
cat("[training set = ", training_set_perc,"%]\n", sep="")
cat("[test set = ", (100-training_set_perc),"%]\n", sep="")

# the training set is the first 60% of the whole dataset
training_set_first_index <- 1 # NEW
training_set_last_index <- round(dim(cancer_data_norm)[1]*training_set_perc/100) # NEW

# the test set is the last 40% of the whole dataset
test_set_first_index <- training_set_last_index+1 # NEW
test_set_last_index <- dim(cancer_data_norm)[1] # NEW

cat("[Creating the training set and test set for the values]\n")
cancer_data_train <- cancer_data_norm[training_set_first_index:training_set_last_index, 1:(target_index)] # NEW
cancer_data_test <- cancer_data_norm[test_set_first_index:test_set_last_index, 1:(target_index)] # NEW

cat("[training set dimensions: ", dim(cancer_data_train)[1], " patients]\n")

cat("[test set dimensions: ", dim(cancer_data_test)[1], " patients]\n")

cat("[Creating the training set and test set for the labels \"1\"-\"0\"]\n")
cancer_data_train_labels <- cancer_data_norm[training_set_first_index:training_set_last_index, target_index] # NEW
cancer_data_test_labels <- cancer_data_norm[test_set_first_index:test_set_last_index, target_index]   # NEW


library(class)
library(gmodels)

cat("\n[Training the random forest classifier on the training set]\n")
rf_new <- randomForest(Metastasis ~ ., data=cancer_data_train, importance=TRUE, proximity=TRUE, ntree=thisNtree)

cat(" rf_new$ntree = ", rf_new$ntree, "\n")

cat("\n[Applying the trained random forest classifier on the test set]\n")
cancer_data_test_PRED <- predict(rf_new, cancer_data_test, type="response")

cancer_data_test_PRED_binary <- as.numeric(cancer_data_test_PRED)

cancer_data_test_PRED_binary[cancer_data_test_PRED_binary>=threshold]=1
cancer_data_test_PRED_binary[cancer_data_test_PRED_binary<threshold]=0

# print(cancer_data_test$class.of.diagnosis)
# print(cancer_data_test_PRED_binary)
# cancer_data_test_PRED_binary

fg_test <- cancer_data_test_PRED[cancer_data_test_labels==1]
bg_test <- cancer_data_test_PRED[cancer_data_test_labels==0]
pr_curve_test <- pr.curve(scores.class0 = fg_test, scores.class1 = bg_test, curve = F)
#plot(pr_curve_test)
# print(pr_curve_test)

cat("[Printing results\n")
confusion_matrix_rates(cancer_data_test_labels, cancer_data_test_PRED_binary)

# mcc_outcome <- mcc(cancer_data_test_labels, cancer_data_test_PRED_binary)



