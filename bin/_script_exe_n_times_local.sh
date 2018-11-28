#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#
set -o nounset -o pipefail -o errexit
set -o xtrace

iteTot=10

i=1
outputFile=""
for i in $( seq 1 $iteTot )
do

  echo $i
  today=`date +%Y-%m-%d`
  random_number=$(shuf -i1-100000 -n1)
  method="random_forest"
  jobName=$method"_"$today"_rand"$random_number
  outputFile="../results/"$jobName

  /usr/bin/Rscript random_forest_class.r > $outputFile 2> $outputFile

done

# i=1
# outputFile=""
# for i in $( seq 1 $iteTot )
# do
# 
#   echo $i
#   today=`date +%Y-%m-%d`
#   random_number=$(shuf -i1-100000 -n1)
#   method="lin_reg"
#   jobName=$method"_"$today"_rand"$random_number
#   outputFile="../results/"$jobName
# 
#   /usr/bin/Rscript lin_reg.r > $outputFile 2> $outputFile
# 
# done
# 
# i=1
# outputFile=""
# for i in $( seq 1 $iteTot )
# do
# 
#   echo $i
#   today=`date +%Y-%m-%d`
#   random_number=$(shuf -i1-100000 -n1)
#   method="svm"
#   jobName=$method"_"$today"_rand"$random_number
#   outputFile="../results/"$jobName
# 
#   /usr/bin/Rscript svm.r > $outputFile 2> $outputFile
# 
# done
# 
# i=1
# outputFile=""
# for i in $( seq 1 $iteTot )
# do
# 
#   echo $i
#   today=`date +%Y-%m-%d`
#   random_number=$(shuf -i1-100000 -n1)
#   method="knn"
#   jobName=$method"_"$today"_rand"$random_number
#   outputFile="../results/"$jobName
# 
#   /usr/bin/Rscript knn.r > $outputFile 2> $outputFile
# 
# done
