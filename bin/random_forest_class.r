setwd(".")
options(stringsAsFactors = FALSE)

list.of.packages <- c("PRROC", "e1071", "randomForest","class", "gmodels")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library("PRROC")
library("e1071")
library("randomForest")
library("class")
library("gmodels")

source("./confusion_matrix_rates.r")
source("./utils.r")

# args = commandArgs(trailingOnly=TRUE)
# thisNtree <- as.integer(args[1])

# thisNtree <- 5000

threshold <- 0.5
fileName <- "../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv"
cancer_data_norm <- read.csv(file=fileName,head=TRUE,sep=",",stringsAsFactors=FALSE)
cat("fileName = ", fileName, "\n", sep="")

cat("[Randomizing the rows]\n")
cancer_data_norm <- cancer_data_norm[sample(nrow(cancer_data_norm)),] # shuffle the rows

totalElements <- dim(cancer_data_norm)[1]

subsets_size <- 10000

target_index <- dim(cancer_data_norm)[2]

target_label <- colnames(cancer_data_norm[target_index])

if (subsets_size != totalElements) {
    cat("ATTENTION: We are running the method on a subset of the original dataset, \n", sep="")
    cat(" containing only ", subsets_size, " elements \n", sep="")
    cat(" instead of ", totalElements, " elements \n", sep="")
}

cancer_data_norm <- cancer_data_norm[1:subsets_size, ]

dataset_dim_retriever(cancer_data_norm)
imbalance_retriever(cancer_data_norm$Metastasis)

training_set_perc=50
INPUT_PERC_POS <- 50
cat("[training set = ", training_set_perc,"%]\n", sep="")
cat("[test set = ", (100-training_set_perc),"%]\n", sep="")

artificialBalance <- TRUE
balancedFlag <- TRUE # flag that sets everything to 50% 50% ratio

if (artificialBalance == TRUE) {


    train_data_balancer_output <- train_data_balancer(cancer_data_norm, target_index, training_set_perc, INPUT_PERC_POS, balancedFlag)

    cancer_data_train <- train_data_balancer_output[[1]]
    cancer_data_test <- train_data_balancer_output[[2]]
    
     # Creating the subsets for the targets
    cancer_data_train_labels <- cancer_data_train[, target_index] # NEW
     cancer_data_test_labels <- cancer_data_test[, target_index]   # NEW

} else {


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

}


dataset_dim_retriever(cancer_data_train)
imbalance_retriever(cancer_data_train$Metastasis)


cat("\n[Training the random forest classifier on the training set]\n")

# rf_new <- randomForest(Metastasis ~ ., data=cancer_data_train, importance=TRUE, proximity=TRUE, ntree=thisNtree)
allFeaturesFormula <- Metastasis ~ .
thisFormulaTop2features <- Metastasis ~ DerivedSS1977 + RXSumm..SurgOthReg.Dis.2003..
thisFormulaTop3features <- Metastasis ~ DerivedSS1977 + RXSumm..SurgOthReg.Dis.2003.. + T
thisFormula_TN <- Metastasis ~ T + N
thisFormula_TNTumorSize <- Metastasis ~ T + N + TumorSize
thisFormula_TNTumorSizeAge <- Metastasis ~ T + N + TumorSize + Age # top predictions among the non-metastasis features

selectedFormula <- thisFormula_TNTumorSizeAge
rf_new <- randomForest(selectedFormula, data=cancer_data_train, importance=TRUE, proximity=TRUE)
cat("\nFeatures used in this prediction: ", toString(selectedFormula), "\n\n", sep="")


cat("\n[Applying the trained random forest classifier on the test set]\n")
cancer_data_test_PRED <- predict(rf_new, cancer_data_test, type="response")

fg_test <- cancer_data_test_PRED[cancer_data_test_labels==1]
bg_test <- cancer_data_test_PRED[cancer_data_test_labels==0]

pr_curve_test <- pr.curve(scores.class0 = fg_test, scores.class1 = bg_test, curve = F)
# plot(pr_curve_test)
print(pr_curve_test)

roc_curve_test <- roc.curve(scores.class0 = fg_test, scores.class1 = bg_test, curve = F)
# plot(pr_curve_test)
print(roc_curve_test)

cancer_data_test_PRED_binary <- as.numeric(cancer_data_test_PRED)
cancer_data_test_PRED_binary[cancer_data_test_PRED_binary>=threshold]=1
cancer_data_test_PRED_binary[cancer_data_test_PRED_binary<threshold]=0

cat("[Printing results\n")
confusion_matrix_rates(cancer_data_test_labels, cancer_data_test_PRED_binary, "@@@ Test set @@@")

# mcc_outcome <- mcc(cancer_data_test_labels, cancer_data_test_PRED_binary)



