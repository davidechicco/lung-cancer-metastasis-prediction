#!/usr/bin/env Rscript

library("ggplot2")
library("ggpubr")

prc_data_norm <- read.csv(file="../data/LungCancerDataset_AllRecords_NORM.csv",head=TRUE,sep=",",stringsAsFactors=FALSE)

ggscatter(prc_data_norm, x = "Stage", y = "Metastasis", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Metastasis", ylab = "Stage")

cat("Pearson correlation between Stage and Metastasis: ")
cor(as.numeric(prc_data_norm$Stage), as.numeric(prc_data_norm$Metastasis), method = c("pearson"))

cat("Kendall correlation between Stage and Metastasis: ")
cor(as.numeric(prc_data_norm$Stage), as.numeric(prc_data_norm$Metastasis), method = c("kendall"))

cat("Spearman correlation between Stage and Metastasis: ")
cor(as.numeric(prc_data_norm$Stage), as.numeric(prc_data_norm$Metastasis), method = c("spearman"))





res <- cor.test(prc_data_norm$Stage, prc_data_norm$Metastasis, 
                    method = "pearson")
res