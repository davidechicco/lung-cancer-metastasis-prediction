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

thisNtree <- 500

threshold <- 0.5

cancer_data_norm <- read.csv(file="../data/LungCancerDataset_AllRecords_NORM_reduced_features.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

cat("[Randomizing the rows]\n")
cancer_data_norm <- cancer_data_norm[sample(nrow(cancer_data_norm)),] # shuffle the rowsps

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
confusion_matrix_rates(cancer_data_test_labels, cancer_data_test_PRED_binary, "@@@ Test set @@@")

# mcc_outcome <- mcc(cancer_data_test_labels, cancer_data_test_PRED_binary)



