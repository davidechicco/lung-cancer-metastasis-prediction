options(stringsAsFactors = FALSE)

# function that prints two decimals of a number
dec_two <- function(x) {
  return (format(round(x, 2), nsmall = 2));
}


# Function that reads in a vector made of binary values and prints the imbalance rates
imbalance_retriever <- function(thisVector)
{
  lun <- length(table(thisVector))
  if (lun != 2) {
  
    print("This vector is not binary. The imbalance_retriever() function will stop here");
    return ;
  
  }  

  cat("\n[Imbalance of this dataset]\n")
  number_of_elements_of_first_class <- unname(table(thisVector)[1])
  name_of_elements_of_first_class <- names(table(thisVector)[1])
  cat("[class: ",name_of_elements_of_first_class, "  #elements = ", number_of_elements_of_first_class, "]\n", sep="")
  cat(dec_two(unname(table(thisVector))[1]*100/length(thisVector)),"%\n", sep="")
  
  number_of_elements_of_second_class <-unname(table(thisVector)[2])
  name_of_elements_of_second_class <-names(table(thisVector)[2])
  cat("[class: ",name_of_elements_of_second_class, "  #elements = ", number_of_elements_of_second_class, "]\n", sep="")
  cat(dec_two(unname(table(thisVector))[2]*100/length(thisVector)),"%\n", sep="")
  
  cat("\n")

}


