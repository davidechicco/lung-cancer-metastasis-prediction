setwd(".")
options(stringsAsFactors = FALSE)

# list.of.packages <- c("PRROC", "e1071", "randomForest","class", "gmodels", "formula.tools")

list.of.packages <- c("PRROC", "e1071", "randomForest","class", "gmodels", "SEERaBomb", "miceadds")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


# library("PRROC")
# library("e1071")
# library("randomForest")
# library("class")
# library("gmodels")
library("SEERaBomb")
library("miceadds")

num_to_return <- 1
exe_num <- sample(1:as.numeric(Sys.time()), num_to_return)

# https://rdrr.io/cran/SEERaBomb/src/inst/doc/examples/mkDataBinaries.R

df <- getFields(seerHome="./SEER_1973_2015_CUSTOM_TEXTDATA/")

rdf=pickFields(df)
# pickFields(df, picks=c("casenum","reg","race","sex","agedx",  "yrbrth","seqnum","modx","yrdx","histo3",      "ICD9","COD","surv","radiatn","chemo", "tvalue"))

## Dataset with date of birth

rdf_TumorSize_Age=pickFields(df, picks=c("casenum","reg","race","sex","agedx",  "yrbrth","seqnum","modx","yrdx","histo3",   "ICD9","COD","surv","radiatn","chemo", "tvalue", "mvalue"))

mkSEER(rdf_TumorSize_Age, seerHome="./SEER_1973_2015_CUSTOM_TEXTDATA/")

load.Rdata("./SEER_1973_2015_CUSTOM_TEXTDATA/mrgd/cancDef_ALL_TumorSize_Metastasis.RData", "data_ALL_TMY")

lung_cancer_data <- data_ALL_TMY%>%filter(cancer=="lung")
lung_cancer_dataframe <- as.data.frame(lung_cancer_data)
colnames(lung_cancer_dataframe)

lung_cancer_dataframe_TMY <- lung_cancer_dataframe[,c("tvalue","yrbrth","mvalue")]

# Removes the NA values
lung_cancer_dataframe_TMY_complete_cases <- lung_cancer_dataframe_TMY[complete.cases(lung_cancer_dataframe_TMY), ]


# AJCC 3rd Edition, TNM, and Stage in SEER Data
# https://seer.cancer.gov/seerstat/variables/seer/ajcc-stage/3rd.html

# Let's keep only M0 (code 00) and M1 (code 10) 
lung_cancer_dataframe_TMY_noNA_onlyM0M1 <- lung_cancer_dataframe_TMY_complete_cases[lung_cancer_dataframe_TMY_complete_cases$"mvalue" == "10" | lung_cancer_dataframe_TMY_complete_cases$"mvalue" == "0", ]

colnames(lung_cancer_dataframe_TMY_noNA_onlyM0M1) <- c("TumorSize", "YearOfBirth", "Metastasis")

 write.table(lung_cancer_dataframe_TMY_noNA_onlyM0M1, paste("lung_cancer_dataframe_TMY_noNA_onlyM0M1_time", exe_num, ".csv", sep=""),col.names=TRUE, row.names=FALSE, sep=",")
 
## Dataset with "agerec"

rdf_TumorSize_AgeB=pickFields(df, picks=c("casenum","reg","race","sex","agedx",  "agerec","seqnum","modx","yrdx","histo3",   "ICD9","COD","surv","radiatn","chemo", "tvalue", "mvalue"))

mkSEER(rdf_TumorSize_AgeB, seerHome="./SEER_1973_2015_CUSTOM_TEXTDATA/", outFile="cancDef_ALL_TumorSize_Age_Metastasis")

load.Rdata("./SEER_1973_2015_CUSTOM_TEXTDATA/mrgd/cancDef_ALL_TumorSize_Age_Metastasis.RData", "data_ALL_TMA")

lung_cancer_age_data <- data_ALL_TMA%>%filter(cancer=="lung")
lung_cancer_age_dataframe <- as.data.frame(lung_cancer_age_data)
colnames(lung_cancer_age_dataframe)

lung_cancer_dataframe_TMA <- lung_cancer_age_dataframe[,c("tvalue","agedx","mvalue")]

# Removes the NA values
lung_cancer_dataframe_TMA_complete_cases <- lung_cancer_dataframe_TMA[complete.cases(lung_cancer_dataframe_TMA), ]


# AJCC 3rd Edition, TNM, and Stage in SEER Data
# https://seer.cancer.gov/seerstat/variables/seer/ajcc-stage/3rd.html

# Let's keep only M0 (code 00) and M1 (code 10) 
lung_cancer_dataframe_TMA_noNA_onlyM0M1 <- lung_cancer_dataframe_TMA_complete_cases[lung_cancer_dataframe_TMA_complete_cases$"mvalue" == "10" | lung_cancer_dataframe_TMA_complete_cases$"mvalue" == "0", ]

colnames(lung_cancer_dataframe_TMA_noNA_onlyM0M1) <- c("TumorSize", "AgeAtDiagnosis", "Metastasis")
lung_cancer_dataframe_TMA_noNA_onlyM0M1$Metastasis <- lung_cancer_dataframe_TMA_noNA_onlyM0M1$Metastasis/10

 write.table(lung_cancer_dataframe_TMA_noNA_onlyM0M1, paste("lung_cancer_dataframe_TMA_noNA_onlyM0M1_time", exe_num, ".csv", sep=""),col.names=TRUE, row.names=FALSE, sep=",")
 

