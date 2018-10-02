# Computational prediction of metastasis of lung cancer patients from clinical features
Machine learning prediction of lung cancer metastasis from clinical data patients

First step: mapping of the feature values and normalization into the [0; 1] interval for each feature

`Rscript normalization.r`

Machine learning methods:

linear regression

`Rscript lin_reg.r`

k-nearest neighbors

`Rscript knn.r`

support vector machines

`Rscript svm.r`

random forest classification (top method)

`Rscript random_forest_class.r`

deep neural network

`th ann_script_val.lua ../data/LungCancerDataset_AllRecords_NORM.csv`

