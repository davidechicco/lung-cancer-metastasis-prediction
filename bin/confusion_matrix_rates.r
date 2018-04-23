options(stringsAsFactors = FALSE)
# library("clusterSim")
library("e1071")
library("PRROC")

# Confusion matrix rates
confusion_matrix_rates <- function (actual, predicted)
{
  
  TP <- sum(actual == 1 & predicted == 1)
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  
  f1_score <- 2*TP / (2*TP + FP + FN)
  accuracy <- (TN+TP) / (TN + TP + FP + FN)
  
  cat("f1_score = ", f1_score, " (worst: 0.0; best: 1.0)\n", sep="")
  cat("accuracy = ", accuracy, " (worst: 0.0; best: 1.0)\n", sep="")
  
  recall <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  cat("\n")
  cat("true positive rate = recall = ", recall, " (worst: 0.0; best: 1.0)\n", sep="")
  cat("true negative rate = specificity = ", specificity, " (worst: 0.0; best: 1.0)\n", sep="")
  cat("\n")

}

# Matthews correlation coefficient
mcc <- function (actual, predicted)
{
  # Compute the Matthews correlation coefficient (MCC) score
  # Jeff Hebert 9/1/2016
  # Geoffrey Anderson 10/14/2016 
  # Added zero denominator handling.
  # Avoided overflow error on large-ish products in denominator.
  #
  # actual = vector of true outcomes, 1 = Positive, 0 = Negative
  # predicted = vector of predicted outcomes, 1 = Positive, 0 = Negative
  # function returns MCC
  
  TP <- sum(actual == 1 & predicted == 1)
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  #TP;TN;FP;FN # for debugging
  sum1 <- TP+FP; sum2 <-TP+FN ; sum3 <-TN+FP ; sum4 <- TN+FN;
  denom <- as.double(sum1)*sum2*sum3*sum4 # as.double to avoid overflow error on large products
  if (any(sum1==0, sum2==0, sum3==0, sum4==0)) {
    denom <- 1
  }
  mcc <- ((TP*TN)-(FP*FN)) / sqrt(denom)
  
  cat("\nMCC = ",mcc, " (worst possible: -1; best possible: +1)\n\n", sep="")
  
  return(mcc)
}