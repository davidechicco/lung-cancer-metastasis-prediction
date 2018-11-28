#!/usr/bin/env Rscript


setwd(".")
options(stringsAsFactors = FALSE)
# library("clusterSim")
library("e1071")
library("PRROC")
tau = 0.5

source("./confusion_matrix_rates.r")


cancer_data_norm <- read.csv(file="../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

cancer_data_norm <- cancer_data_norm[sample(nrow(cancer_data_norm)),] # shuffle the rows

target_index <- dim(cancer_data_norm)[2]

training_set_perce <- 80

cat("training_set_perce = ", training_set_perce, "%\n", sep="")

# the training set is the first training_set_perce% of the whole dataset
training_set_first_index <- 1 # NEW
training_set_last_index <- round(dim(cancer_data_norm)[1]*training_set_perce/100) # NEW

 # the test set is the last 20% of the whole dataset
test_set_first_index <- round(dim(cancer_data_norm)[1]*training_set_perce/100)+1 # NEW
test_set_last_index <- dim(cancer_data_norm)[1] # NEW

cat("[Creating the subsets for the values]\n")
prc_data_train <- cancer_data_norm[training_set_first_index:training_set_last_index, 1:(target_index-1)] # NEW
prc_data_test <- cancer_data_norm[test_set_first_index:test_set_last_index, 1:(target_index-1)] # NEW


cat("[Creating the subsets for the labels \"1\"-\"0\"]\n")
prc_data_train_labels <- cancer_data_norm[training_set_first_index:training_set_last_index, target_index] # NEW
prc_data_test_labels <- cancer_data_norm[test_set_first_index:test_set_last_index, target_index]   # NEW

library(class)
library(gmodels)

# apply k-NN with k_best to the test set

cat("\n[Training the linear regression model on training set & applying the linear regression to test set]\n", sep="")

lin_reg_model_new <- lm(prc_data_train_labels ~ ., data=prc_data_train)
prc_data_test_pred <- predict(lin_reg_model_new, prc_data_test)

prc_data_test_pred_bin <- as.numeric(prc_data_test_pred)
prc_data_test_pred_bin[prc_data_test_pred_bin>=tau]<-1
prc_data_test_pred_bin[prc_data_test_pred_bin<tau]<-0

confusion_matrix_rates(prc_data_test_labels, prc_data_test_pred_bin, "@@@ Test set @@@")


