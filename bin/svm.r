setwd(".")
options(stringsAsFactors = FALSE)
#library("clusterSim")
library("e1071")
library("PRROC")
tau = 0.5

source("./confusion_matrix_rates.r")

cancer_data_norm <- read.csv(file="../data/LungCancerDataset_AllRecords_NORM_reduced_features.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

cancer_data_norm <- cancer_data_norm[sample(nrow(cancer_data_norm)),] # shuffle the rows

target_index <- dim(cancer_data_norm)[2]

# the training set is the first 60% of the whole dataset
training_set_first_index <- 1 # NEW
training_set_last_index <- round(dim(cancer_data_norm)[1]*60/100) # NEW

# the validation set is the following 20% of the whole dataset
validation_set_first_index <- round(dim(cancer_data_norm)[1]*60/100)+1 # NEW
validation_set_last_index <- round(dim(cancer_data_norm)[1]*80/100) # NEW

# the test set is the last 20% of the whole dataset
test_set_first_index <- round(dim(cancer_data_norm)[1]*80/100)+1 # NEW
test_set_last_index <- dim(cancer_data_norm)[1] # NEW

cat("[Creating the subsets for the values]\n")
cancer_data_train <- cancer_data_norm[training_set_first_index:training_set_last_index, 1:(target_index-1)] # NEW
cancer_data_validation <- cancer_data_norm[validation_set_first_index:validation_set_last_index, 1:(target_index-1)] # NEW
cancer_data_test <- cancer_data_norm[test_set_first_index:test_set_last_index, 1:(target_index-1)] # NEW



cat("[Creating the subsets for the labels \"1\"-\"0\"]\n")
cancer_data_train_labels <- cancer_data_norm[training_set_first_index:training_set_last_index, target_index] # NEW
cancer_data_validation_labels <- cancer_data_norm[validation_set_first_index:validation_set_last_index, target_index] # NEW
cancer_data_test_labels <- cancer_data_norm[test_set_first_index:test_set_last_index, target_index]   # NEW

library(class)
library(gmodels)

# The k value must be lower than the size of the training set
maxK <- 10 #NEW

mcc_array <- character(length(maxK))

# NEW PART:

c_array = c(0.001, 0.01, 0.1, 1, 10)
mccCounter = 1

cat("\n[Optimization of the hyper-parameter C start]\n")
# optimizaion loop
for(thisC in c_array)
{
  # apply k-NN with the current K value
  # train on the training set, evaluate in the validation set by computing the MCC
  # save the MCC corresponding to the current K value
  
  cat("[Training the SVM model (with C=",thisC,") on training set & applying the SVM model to validation set]\n", sep="")
  
  svm_model <- svm(cancer_data_train_labels ~ ., cost=thisC, data=cancer_data_train, method = "C-classification", kernel = "linear")
    
  cancer_data_validation_pred <- predict(svm_model, cancer_data_validation)
  
  # CrossTable(x=cancer_data_validation_labels, y=cancer_data_validation_pred, prop.chisq=FALSE)
  
  cancer_data_validation_labels_binary_TEMP <- replace(cancer_data_validation_labels, cancer_data_validation_labels=="M", 1)
  cancer_data_validation_labels_binary <- replace(cancer_data_validation_labels_binary_TEMP, cancer_data_validation_labels=="B", 0)
  cancer_data_validation_labels_binary <- as.numeric (cancer_data_validation_labels_binary)
  # cancer_data_validation_labels_binary
  
  cancer_data_validation_pred_AS_CHAR <- as.character(cancer_data_validation_pred)
  cancer_data_validation_pred_binary_TEMP <- replace(cancer_data_validation_pred_AS_CHAR, cancer_data_validation_pred_AS_CHAR=="M", 1)
  cancer_data_validation_pred_binary <- replace(cancer_data_validation_pred_binary_TEMP, cancer_data_validation_pred_AS_CHAR=="B", 0)
  cancer_data_validation_pred_binary <- as.numeric (cancer_data_validation_pred_binary)
  
  cancer_data_validation_pred_binary[cancer_data_validation_pred_binary>=tau]<-1
  cancer_data_validation_pred_binary[cancer_data_validation_pred_binary<tau]<-0
  
  # cancer_data_validation_pred_binary
#   
#   fg <- cancer_data_validation_pred[cancer_data_validation$Biopsy==1]
#   bg <- cancer_data_validation_pred[cancer_data_validation$Biopsy==0]
#   pr_curve <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = F)
#   print(pr_curve)
  
  mcc_outcome <- mcc(cancer_data_validation_labels_binary, cancer_data_validation_pred_binary)
  cat("When C=",thisC,", the MCC value is ",mcc_outcome, "\t (worst possible: -1; best possible: +1)\n", sep="")
  
  mcc_array[mccCounter] <- mcc_outcome
  mccCounter = mccCounter + 1
}

# select the k corresponding to the highest MCC and call it k_best
bestMCC <- max(mcc_array)
bestCindex <- match(bestMCC, mcc_array)
cat("\nThe best C value is ", c_array[bestCindex],", corresponding to MCC=", mcc_array[bestCindex],"\n", sep="")

cat("[Optimization end]\n\n")

cat("\n @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ \n")

# apply k-NN with k_best to the test set

cat("\n[Training the SVM model (with the OPTIMIZED hyper-parameter C=",c_array[bestCindex],") on training set & applying the SVM to the test set]\n", sep="")
#cancer_data_test_pred <- knn(train = cancer_data_train, test = cancer_data_test, cl = cancer_data_train_labels, k=bestK)

svm_model_new <- svm(cancer_data_train_labels ~ ., cost=c_array[bestCindex], data=cancer_data_train, method = "C-classification", kernel = "linear")
cancer_data_test_pred <- predict(svm_model_new, cancer_data_test)

cancer_data_test_labels_binary_TEMP <- replace(cancer_data_test_labels, cancer_data_test_labels=="M", 1)
cancer_data_test_labels_binary <- replace(cancer_data_test_labels_binary_TEMP, cancer_data_test_labels=="B", 0)
cancer_data_test_labels_binary <- as.numeric (cancer_data_test_labels_binary)
# cancer_data_test_labels_binary

cancer_data_test_pred_AS_CHAR <- as.character(cancer_data_test_pred)
cancer_data_test_pred_binary_TEMP <- replace(cancer_data_test_pred_AS_CHAR, cancer_data_test_pred_AS_CHAR=="M", 1)
cancer_data_test_pred_binary <- replace(cancer_data_test_pred_binary_TEMP, cancer_data_test_pred_AS_CHAR=="B", 0)
cancer_data_test_pred_binary <- as.numeric (cancer_data_test_pred_binary)

cancer_data_test_pred_binary[cancer_data_test_pred_binary>=tau]<-1
cancer_data_test_pred_binary[cancer_data_test_pred_binary<tau]<-0
# cancer_data_test_pred_binary
# 
# fg_test <- cancer_data_test_pred[cancer_data_test$Biopsy==1]
# bg_test <- cancer_data_test_pred[cancer_data_test$Biopsy==0]
# pr_curve_test <- pr.curve(scores.class0 = fg_test, scores.class1 = bg_test, curve = F)
# plot(pr_curve_test)
# 
# print(pr_curve_test)

# mcc_outcome <- mcc(cancer_data_test_labels_binary, cancer_data_test_pred_binary)

confusion_matrix_rates(cancer_data_test_labels_binary, cancer_data_test_pred_binary, "@@@ Test set @@@")


