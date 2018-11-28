setwd(".")
options(stringsAsFactors = FALSE)
# library("clusterSim")

library("OneR");
library(class)
library(gmodels)
source("./confusion_matrix_rates.r")

threshold <- 0.5

fileName <- "../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv"
prc_data_norm <- read.csv(file=fileName, head=TRUE,sep=",",stringsAsFactors=FALSE)

cat("fileName: ", fileName, sep="")

prc_data_norm <- prc_data_norm[sample(nrow(prc_data_norm)),] # shuffle the rows

target_index <- dim(prc_data_norm)[2]

training_set_perce = 80
cat("training_set_perce = ", training_set_perce, "\n", sep="")

# the training set is the first 60% of the whole dataset
training_set_first_index <- 1 # NEW
training_set_last_index <- round(dim(prc_data_norm)[1]*training_set_perce/100) # NEW

# the test set is the last 40% of the whole dataset
test_set_first_index <- training_set_last_index+1 # NEW
test_set_last_index <- dim(prc_data_norm)[1] # NEW

cat("[Creating the subsets for the values]\n")
prc_data_train <- prc_data_norm[training_set_first_index:training_set_last_index, 1:(target_index)] # NEW
prc_data_test <- prc_data_norm[test_set_first_index:test_set_last_index, 1:(target_index)] # NEW

prc_data_test_labels  <- prc_data_norm[test_set_first_index:test_set_last_index, target_index]   # NEW


print("dim(prc_data_train)")
print(dim(prc_data_train))

print("dim(prc_data_test)")
print(dim(prc_data_test))


# #rf_new <- randomForest(Metastasis ~ ., data=prc_data_train, importance=TRUE, proximity=TRUE)


# Original application of One Rule with all the dataset
prc_model_train <- OneR(prc_data_train, verbose = TRUE)

# Generation of the CART model
# prc_model_train <- OneR(Metastasis ~ keep.side + platelet.count..PLT., method="class", data=prc_data_train);

summary(prc_model_train)
prediction <- predict(prc_model_train, prc_data_test)
# eval_model(prediction, prc_data_test)

prediction_binary <- as.numeric(prediction) -1
prc_data_test_PRED_binary <- data.frame(prediction)

confusion_matrix_rates(prc_data_test_labels, prediction_binary, "@@@ Test set @@@")



