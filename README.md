# Computational prediction of metastasis of lung cancer patients from clinical features
Machine learning prediction of lung cancer metastasis from clinical data patients

First step: mapping of the feature values and normalization into the [0; 1] interval for each feature

`Rscript normalization.r`

Machine learning methods, instructions on Linux Ubuntu:

linear regression

`/usr/bin/Rscript lin_reg.r`

k-nearest neighbors

`/usr/bin/Rscript  knn.r`

support vector machines

`/usr/bin/Rscript  svm.r`

decision tree

`/usr/bin/Rscript  cart.r`

one rule

`/usr/bin/Rscript  oner_class.r`

naive bayes

`/usr/bin/Rscript  naive_bayes.r`

random forest classification (top method)

`/usr/bin/Rscript  random_forest_class.r`

deep neural network

`th ann_script_val.lua`

## Question on Medical Science Stack Exchange
[Can tumor size (T) and presence of cancer in the lymph nodes (N) in patients with lung cancer be identified on the first visit?](https://medicalsciences.stackexchange.com/questions/18040/can-tumor-size-t-and-presence-of-cancer-in-the-lymph-nodes-n-in-patients-wit)
