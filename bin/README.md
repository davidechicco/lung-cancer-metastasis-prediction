
# Computational prediction of patients diagnosis applied to a cervical cancer dataset
Computational prediction of patients diagnosis applied to a cervical cancer dataset

## Installation
To run the scripts, you need to have installed:
* **R** (version 3.3.2)
* R packages **rgl**, **clusterSim** and **PRROC**
* **git** (version 1.8.3.1)
* **Torch** (version 7)
* **LuaRocks** (version 2.3.0)

You need to have root privileges, an internet connection, and at least 1 GB of free space on your hard disk. We here provide the instructions to install all the needed programs and dependencies on Linux CentOS, Linux Ubuntu, and Mac OS. Our scripts were originally developed on a Linux Ubuntu computer.

### Dependency installation for Linux Ubuntu
<img src="http://www.internetpost.it/wp-content/uploads/2016/04/ubuntu-head.png" width="150" align="right">
Here are the instructions to install all the programs and libraries needed by our scripts on a Linux Ubuntu computer, from a shell terminal. We tested these instructions on a Dell Latitude 3540 laptop, running Linux Ubuntu 16.10 operating system, and having a 64-bit kernel, in February 2017. If you are using another operating system version, some instructions might be slightly different.

Install R and its rgl, clusterSim, PRROC packages:<br>
`sudo apt-get -y install r-base-core`<br>
`sudo apt-get -y install r-cran-rgl`<br>
`sudo Rscript -e 'install.packages(c("rgl", "clusterSim", "PRROC"), repos="https://cran.rstudio.com")'`<br>

Install git:<br>
`sudo apt-get -y install git`<br>

Install Torch and luarocks:<br>
`# in a terminal, run the commands WITHOUT sudo`<br>
`git clone https://github.com/torch/distro.git ~/torch --recursive`<br>
`cd ~/torch; bash install-deps;`<br>
`./install.sh`<br>

`source ~/.bashrc`<br>
`cd ~`<br>

`sudo apt-get -y install luarocks`<br>
`sudo luarocks install csv`<br>

Clone this repository:<br>
`git clone https://github.com/davidechicco/cervical_cancer_predictions.git`<br>

Move to the project main directory, and download the mesothelioma dataset file:<br>
`cd /cervical_cancer_predictions/` <br>

### Dependency installation for Linux CentOS
<img src="http://brettspence.com/wp-content/uploads/2014/11/centos-7-logo-580x118.jpg" width="100" align="right">
Here are the instructions to install all the programs and libraries needed by our scripts on a Linux CentOS computer, from a shell terminal. We tested these instructions on a Dell Latitude 3540 laptop, running Linux Ubuntu 16.10 operating system, and having a 64-bit kernel, in February 2017. If you are using another operating system version, some instructions might be slightly different.

Install R, its dependencies, and is rgl, clusterSim, randomForest packages:<br>
`sudo yum -y install R` <br>
`sudo yum -y install mesa-libGL` <br>
`sudo yum -y  install mesa-libGL-devel` <br>
`sudo yum -y  install mesa-libGLU` <br>
`sudo yum -y  install mesa-libGLU-devel` <br>
`sudo yum -y install libpng-devel` <br>
`sudo Rscript -e 'install.packages(c("rgl", "clusterSim", "PRROC"), repos="https://cran.rstudio.com")'` <br>

Install Torch and luarocks:<br>
`sudo apt-get -y install git` <br>
`# in a terminal, run the commands WITHOUT sudo` <br>
`git clone https://github.com/torch/distro.git ~/torch --recursive` <br>
`cd ~/torch; bash install-deps;` <br>
`./install.sh` <br>

`source ~/.bashrc`<br>
`cd ~`<br>

`sudo yum -y install luarocks` <br>
`sudo luarocks install csv` <br>

Clone this repository:<br>
`git clone https://github.com/davidechicco/cervical_cancer_predictions.git`<br>

Move to the project main directory, and download the mesothelioma dataset file:<br>
`cd /cervical_cancer_predictions/` <br>

### Dependency installation for Mac OS
<img src="https://www.technobuffalo.com/wp-content/uploads/2015/06/Mac-OS-logo.jpg" width="150" align="right">
Here are the instructions to install all the programs and libraries needed by our scripts on a Mac computer, from a shell terminal. We tested these instructions on an Apple computer running a Mac OS macOS 10.12.2 Sierra operating system, in March 2017. If you are using another operating system version, some instructions might be slightly different.

Manually download and install XQuartz from https://www.xquartz.org <br>

Install R and its packages:<br>
`brew install r`<br>
`sudo Rscript -e 'install.packages(c("rgl”, "clusterSim”, "PRROC), repos="https://cran.rstudio.com")' `<br>

Install the development tools (such as gcc):<br>
`xcode-select --install`<br>

Install Torch and laurocks:<br>
`git clone https://github.com/torch/distro.git ~/torch --recursive`<br>
`cd ~/torch; bash install-deps`<br>
`./install.sh`<br>
`cd ~`<br>

`brew install lua`<br>
`source ~/.profile`<br>
`sudo luarocks install csv`<br>

Clone this repository:<br>
`git clone https://github.com/davidechicco/cervical_cancer_predictions.git`<br>

# Execution
A deep learning method for prediction of cervical cancer diagnoses from risk factors

This code implements a deep artificial neural network which uses a rectifier linear unit (ReLU) as activaction function, optimizes the hyper-parameters of hidden units and hidden layers, has learning rate = 0.01, iterations = 200 (this values are not final: you're welcome to test alternative ones). Dropout, Xavier initialization, and momentum can be turned on.

The program reads a dataset of profiles of cervical cancer patients (858 patients-rows * 33 features-columns), downloaded from the [University of California Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29). The binary target to predict is the last column on the right ("Biopsy").

The program splits the input dataset into three subsets: training set 60%, validation set 20%, test set 20%. The data instances for each subset are chosen randomly by the program. The program then starts a loop for the optimization of the hyper-parameters: for each hyper-parameter value, it trains the neural network model on the training set, appies the trained model to the validation set, and saves its result (measured with the Matthews correlation coefficient (MCC)). At the end of the loop, the program selects the model who obtained the best MCC, and applies it to the held-out test test. That is the last test.

To run the program on your Linux machine, install Torch and then type:

`th cervical_ann_script_val.lua cervical_arranged_NORM.csv`

A new version with 10-fold cross validation is available. The k-fold cross validation is applied to 80% of the dataset, which is split to training set and validation set at each iteration. Finally, the best trained model is applied to the test set.
To run the script with the k-fold cross validation, type:

`th cervical_ann_script_val_kfold.lua cervical_arranged_NORM.csv`

Other methods (linear regression, k-nearest neighbors, support vector machine):


`Rscript lin_reg.r cervical_arranged_NORM.csv`

`Rscript knn.r cervical_arranged_NORM.csv`

`Rscript svm.r cervical_arranged_NORM.csv`


## Contacts
This sofware was developed by [Davide Chicco](http://www.DavideChicco.it) at [the Princess Margaret Cancer Centre](http://www.uhn.ca/PrincessMargaret/Research/) (Toronto, Ontario, Canada).

For questions or help, please write to davide.chicco(AT)davidechicco.it
