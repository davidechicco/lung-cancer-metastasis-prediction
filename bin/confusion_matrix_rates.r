options(stringsAsFactors = FALSE)
# library("clusterSim")
library("e1071")
library("PRROC")
source("./utils.r")

# Confusion matrix rates
confusion_matrix_rates <- function (actual, predicted, keyword)
{
  
  TP <- sum(actual == 1 & predicted == 1)
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  
  
  cat("\nTOTAL:\n\n")
  cat(" FN = ", (FN), " / ", (FN+TP), "\t (truth == 1) & (prediction < threshold)\n");
  cat(" TP = ", (TP), " / ", (FN+TP),"\t (truth == 1) & (prediction >= threshold)\n\n");
	

  cat(" FP = ", (FP), " / ", (FP+TN), "\t (truth == 0) & (prediction >= threshold)\n");
  cat(" TN = ", (TN), " / ", (FP+TN), "\t (truth == 0) & (prediction < threshold)\n\n");
  
  sum1 <- TP+FP; sum2 <-TP+FN ; sum3 <-TN+FP ; sum4 <- TN+FN;
  denom <- as.double(sum1)*sum2*sum3*sum4 # as.double to avoid overflow error on large products
  if (any(sum1==0, sum2==0, sum3==0, sum4==0)) {
    denom <- 1
  }
  mcc <- ((TP*TN)-(FP*FN)) / sqrt(denom)
  
  f1_score <- 2*TP / (2*TP + FP + FN)
  accuracy <- (TN+TP) / (TN + TP + FP + FN)
  recall <- TP / (TP + FN)
  specificity <- TN / (TN + FP)

  cat("\n\n",keyword,"\t MCC \t F1_score \t accuracy \t TP_rate \t TN_rate \n")
  cat(keyword,"\t", dec_two(mcc), " \t ", dec_two(f1_score), " \t ", dec_two(accuracy), " \t ", dec_two(recall), " \t ", dec_two(specificity), "\n\n")
 
 
#   cat("\nMCC = ", dec_two(mcc), "\n\n", sep="")
#   
#   cat("f1_score = ", dec_two(f1_score), "\n", sep="")
#   cat("accuracy = ", dec_two(accuracy), "\n", sep="")
#   
#   cat("\n")
#   cat("true positive rate = recall = ", dec_two(recall), "\n", sep="")
#   cat("true negative rate = specificity = ", dec_two(specificity), "\n", sep="")
#   cat("\n")

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
  
  cat("\nMCC = ", dec_two(mcc), "\n\n", sep="")
  
  return(mcc)
}
