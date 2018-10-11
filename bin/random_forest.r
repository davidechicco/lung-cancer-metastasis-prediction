

library("randomForest");

fileName <- "../data/LungCancerDataset_AllRecords_NORM_reduced_features.csv"
lung_cancer_datatable <- read.csv(fileName, header = TRUE, sep =",", stringsAsFactors = FALSE);

# lung_cancer_datatable$"Typeoffollow.upexpected" <- NULL
# lung_cancer_datatable$"SiterecwithKaposiandmesothelioma" <- NULL
# 
# lung_cancer_datatable$Metastasis <- lung_cancer_datatable$M
# lung_cancer_datatable$M <- NULL

rf_output <- randomForest(Metastasis ~ ., data=lung_cancer_datatable, importance=TRUE, proximity=TRUE)

dd <- as.data.frame(rf_output$importance);
dd_sorted <- dd[order(dd$"%IncMSE"), ]

print(dd_sorted);

varImpPlot(rf_output)

