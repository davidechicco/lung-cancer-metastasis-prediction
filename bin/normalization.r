# install.packages("class")
# install.packages("gmodels")

# function that normalizes
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) 
}
  

# function that converts a string 
# https://stats.stackexchange.com/a/17995
fromStringToNumeric <- function(x_array) {

   new_x <- as.factor(x_array)
   levels(new_x) <- 1:length(levels(new_x))
   new_x_num <- as.numeric(new_x)

   return (new_x_num)
}


cat("[Reading the data file]\n")
lung_cancer_data <- read.csv("../data/LungCancerDataset_AllRecords.csv", stringsAsFactors = FALSE) 


num_of_columns_original <- dim(lung_cancer_data)[2]
num_of_instances <- dim(lung_cancer_data)[1]
num_of_features_original <- num_of_columns_original - 1

lung_cancer_data_original <- lung_cancer_data

lung_cancer_data$Metastasis <- lung_cancer_data$M
lung_cancer_data$M <- NULL


print("M0 = 0 = tumor NOT spread to distant organs")
print("M1 = 1 = tumor spread to distant organs")

table(lung_cancer_data$Metastasis)  # it helps us to get the numbers of patients
lung_cancer_data$Metastasis <- factor(lung_cancer_data$Metastasis, levels = c("M0", "M1"), labels = c("0", "1"))

lung_cancer_data$Metastasis <- as.numeric(lung_cancer_data$Metastasis)-1

colnames(lung_cancer_data)

lung_cancer_data_num <- lung_cancer_data

# Le's remove this feature that has only one value
#lung_cancer_data$SiterecwithKaposiandmesothelioma <- NULL 

j = 1
for(i in 1:(num_of_columns_original))
{
  if (table(lung_cancer_data[i])==num_of_instances) {
  
    cat("The column ", colnames(lung_cancer_data[i]), "[",i,"] has only one value so will be deleted\n");
    lung_cancer_data_num[j] <- NULL
    j = j - 1
  }
  j = j + 1
}

lung_cancer_data_num$Stage <- NULL

num_of_columns <- dim(lung_cancer_data_num)[2]
num_of_features <- num_of_columns - 1

target_column_index <- grep("Metastasis", colnames(lung_cancer_data_num))

cat("num_of_features = ", num_of_features, "\n")
cat("the target is lung_cancer_data_num$Metastasis, column index =", target_column_index, "\n")

for(i in 1:(num_of_features))
{
  lung_cancer_data_num[,i] <- fromStringToNumeric(lung_cancer_data_num[,i])
}
lung_cancer_data_num$Metastasis <- lung_cancer_data$Metastasis
# lung_cancer_data_num <- lung_cancer_data_num[sample(nrow(lung_cancer_data_num)),] # shuffle the rows


round(prop.table(table(lung_cancer_data_num$Metastasis)) * 100, digits = 1)  # it gives the result in the percentage form rounded of to 1 decimal place( and so it’s digits = 1)

cat("[Normalizing the values of the data file (except the Metastasis target column)]\n")
lung_cancer_data_norm <- as.data.frame(lapply(lung_cancer_data_num[1:num_of_features], normalize))
lung_cancer_data_norm$Metastasis <- lung_cancer_data_num$Metastasis

colnames(lung_cancer_data_norm)

write.table(lung_cancer_data_norm, file = "../data/LungCancerDataset_AllRecords_NORM.csv", row.names=FALSE, na="", col.names=TRUE, sep=",")

