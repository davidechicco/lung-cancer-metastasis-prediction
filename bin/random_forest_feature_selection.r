    
list.of.packages <- c( "randomForest")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library("randomForest");


fileName <- "../data/LungCancerDataset_AllRecords_NORM_27reduced_features.csv"
lung_cancer_datatable <- read.csv(fileName, header = TRUE, sep =",");


cat("[Randomizing the rows]\n")
lung_cancer_random <- lung_cancer_datatable[sample(nrow(lung_cancer_datatable)),] # shuffle the rows

subsets_size <- 1000
lung_cancer_subset <- lung_cancer_random[1:subsets_size, ]

rf_output <- randomForest(Metastasis ~ ., data=lung_cancer_subset, importance=TRUE, proximity=TRUE)

selected_features <- as.data.frame(rf_output$importance);
selected_features_MSE_sorted <- selected_features[order(selected_features$"%IncMSE"), "%IncMSE", drop=FALSE]
print(selected_features_MSE_sorted)

selected_features_Gini_sorted <- selected_features[order(selected_features$"IncNodePurity"), "IncNodePurity", drop=FALSE]
print(selected_features_Gini_sorted)                                                        


varImpPlot(rf_output)

selected_features_MSE_sorted$pos <- seq.int(nrow(selected_features_MSE_sorted))
selected_features_Gini_sorted$pos <- seq.int(nrow(selected_features_Gini_sorted))

selected_features_MSE_sorted$feature <-  rownames(selected_features_MSE_sorted)
selected_features_Gini_sorted$feature <-  rownames(selected_features_Gini_sorted)

colnames(selected_features_MSE_sorted) <- c("MSE", "posMSE", "feature")
colnames(selected_features_Gini_sorted) <- c("Gini", "posGini", "feature")

selected_features_MSE_alphasorted <- selected_features_MSE_sorted[order(selected_features_MSE_sorted$"feature"), ]
print(selected_features_MSE_alphasorted)
selected_features_Gini_alphasorted <- selected_features_Gini_sorted[order(selected_features_Gini_sorted$"feature"), ]
print(selected_features_Gini_alphasorted)

rownames(selected_features_MSE_alphasorted) <- c()
rownames(selected_features_Gini_alphasorted) <- c()

merged_datatable <- merge(selected_features_MSE_alphasorted, selected_features_Gini_alphasorted, by="feature")

merged_datatable$summedRanks <- merged_datatable$posMSE + merged_datatable$posGini

final_sorted_merged_datatable <- merged_datatable[order(-merged_datatable$"summedRanks"), ]

print(final_sorted_merged_datatable$feature)
