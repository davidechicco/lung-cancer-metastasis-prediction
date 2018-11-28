setwd(".")
options(stringsAsFactors = FALSE)

list.of.packages <- c("PRROC", "e1071", "clusterSim","rpart")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library("clusterSim")
library("PRROC")
library("e1071")
library("rpart")

source("./confusion_matrix_rates.r")
source("./utils.r")



threshold <- 0.5

fileName <- "../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv"
mesothelioma_datatable <- read.csv(fileName, header = TRUE, sep =",");

target_index <- dim(mesothelioma_datatable)[2]

cat("fileName: ", fileName, "\n", sep="")

original_mesothelioma_datatable <- mesothelioma_datatable

# shuffle the rows
mesothelioma_datatable <- original_mesothelioma_datatable[sample(nrow(original_mesothelioma_datatable)),] 

# Allocation of the size of the training set
perce_training_set <- 80
size_training_set <- round(dim(mesothelioma_datatable)[1]*(perce_training_set/100))

cat("perce_training_set = ",perce_training_set,"%", sep="")

# Allocation of the training set and of the test set
training_set <- (mesothelioma_datatable[1:size_training_set,])
test_set_index_start <- size_training_set+1
test_set_index_end <- dim(mesothelioma_datatable)[1]
test_set  <- mesothelioma_datatable[test_set_index_start:test_set_index_end,]

test_labels <- mesothelioma_datatable[test_set_index_start:test_set_index_end, target_index]   # NEW


print("dim(training_set)")
print(dim(training_set))

print("dim(test_set)")
print(dim(test_set))


# Generation of the CART model
# cart_model <- rpart(class.of.diagnosis ~ keep.side + platelet.count..PLT., method="class", data=training_set);
cart_model <- rpart(Metastasis ~ ., method="class", data=training_set);

pred_test_predictions <- as.numeric(predict(cart_model, test_set, typ="class"))-1
pred_test_set_labels <- as.numeric(test_set$Metastasis)

prc_data_test_PRED_binary <- as.numeric(pred_test_predictions)

prc_data_test_PRED_binary[prc_data_test_PRED_binary>=threshold]=1
prc_data_test_PRED_binary[prc_data_test_PRED_binary<threshold]=0
# mcc_outcome <- mcc(pred_test_set_labels, prc_data_test_PRED_binary)
# confusion_matrix_rates(pred_test_set_labels, prc_data_test_PRED_binary)

confusion_matrix_rates(test_labels, pred_test_predictions, "@@@ Test set @@@")

