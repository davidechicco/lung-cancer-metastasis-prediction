#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#
set -o nounset -o pipefail -o errexit
# set -o xtrace

iteTot=10

i=1
for i in $( seq 1 $iteTot )
do

  echo $i

  today=`date +%Y-%m-%d`
  random_number=$(shuf -i1-100000 -n1)
  method="random_forest_class"
  jobName=$method"_"$today"_rand"$random_number
  outputFile="../results/"$jobName

  qsub -q hoffmangroup -N "-"$jobName -cwd -b y -o $outputFile -e $outputFile Rscript random_forest_class.r > $outputFile 2> $outputFile

done
