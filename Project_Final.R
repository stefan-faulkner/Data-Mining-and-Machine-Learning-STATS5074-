

#Data Mining and Machine Learning I Project
#Student ID: 2514007


#Installling Packages (In the eventt it is not installed on the grader's machine)
install.packages('class')
install.packages('tidyverse')
install.packages('ggplot2')
install.packages('GGally')
install.packages('corrplot')
install.packages('e1071')
install.packages('ROCR')
install.packages('pROC')
install.packages('rpart')
install.packages('randomForest')
install.packages('rpart.plot')




#Loading libraries
library(tidyverse)
library(ggplot2)
library(GGally)
library(corrplot)
library(class)
library(e1071)
library(ROCR)
library(pROC)
library(rpart)
library(randomForest)
library(rpart.plot)

#Loading Data

#This is done by uploading my specifc dataset to google drive where 
#it is made public and load the url file to where it is located.


id<-  "1QLWOADlM8hLpFl7eBfY-q8C2Be9xZ1lr"  #google file ID
drug_data<- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", id))

drug_data <- drug_data[-1]  #Removing the first column of our dataset since it is irrelevant




str(drug_data)   #Looking on our various data types

#We notice our dependent variable data type is of type'int' from running the command above this will cause a problem further down when building our models , specifically our SVM model

drug_data$Class <-as.factor(drug_data$Class)  #Change dependent variable type to 'factor'


colnames(drug_data)   #Familiarizing with our column names in our drug dataset



#Exploratory Analysis on training data 


#Selected only few predictors since the plot would get very huge
ggpairs(drug_data, 
columns=c("Age" ,"Education" ,"X.Country" ,"Ethnicity" ,"Nscore"   , "Escore" ,"Oscore" , "Ascore","Cscore"),
upper=list(continuous="points")
)  # The above chart is a nice diagnostic of each variable by itself and their relationship with other.




new_df<- drug_data[-12]   #New drug dataframe without dependent variable  --THis is because it is of type factor
correlation<-cor(new_df)
par(mfrow=c(1, 1))
corrplot(correlation,method="color")

#For the plot above, I wanted to have an idea of the correlation between our explanatory variables. 
#This is known as the correlation plot. 
#We can see that Age is somewhat sportively correlated with Education and X.country. 

## We will talk more about this in the project report.



set.seed(1997)   #To ensure reproducility of results 

#Splitting the data 50% training, 25% validation and 25% test

n<- nrow(drug_data)
train<- sample(c(1:n),n/2)
valid<- sample(c(1:n)[-train], n/4)  #Ensure we not using back data from train variable

#Using the variables now to create our training,testing and validation data

train_data <- drug_data[train,]
valid_data <- drug_data[valid, ]
test_data <- drug_data[-c(train,valid),]  #Note that we are using the remaining 25% by just removing the indices what we have defined for the training and validation


dim(drug_data)  #Displaying initial drug data dimensions 



dim(train_data)


dim(valid_data)


dim(test_data)



#We are going to sart with building our k-nearest neighbours (knn) model

#state in project report
# The idea is to evaluate their performance on the validation data to select the optimal value of k and then
# give an estimate of the future performance for a model with the optimal k using the test data.





#Trying to find the optimal  value of K and look at the prediction results on the validation data to choose the k that gives us the best performance

corr.class.rate<-numeric(30)   
for(k in 1:30)
{
  pred.class<-knn(train_data[,-12], valid_data[,-12], train_data[,12], k=k)
  corr.class.rate[k]<-sum((pred.class==valid_data$Class))/length(pred.class)
}


plot(c(1:30),corr.class.rate,type="l",
     main=" Classification Rates",
     xlab="k",ylab="Correct Classification Rate",cex.main=0.7)


corr.class.rate    ##showing the perfromance on all  of the different k values we iterad through

which.max(corr.class.rate)

pred_knn<-corr.class.rate[7]   #SHowing the highest accuracy we had  on our validation data for k=7
pred_knn




#Now we are going to start building our next classification model namely, Support Vector Machines

# We first start with fitting the classification SVM for different values of C
# and calculate the validation prediction error

pred.error<-function(pred,truth)
{
  1-sum(diag(table(pred,truth)))/length(truth)
}


C.val<-c(0.1,0.2,0.5,1,2,3,5,7,8,10)
C.error<-numeric(length(C.val))


for(i in 1:length(C.val))
{
  model<-svm(Class~.,data=train_data,type="C-classification",kernel="linear",
             cost=C.val[i])
  pred.model<-predict(model, valid_data)
  C.error[i]<-pred.error(pred.model,valid_data$Class)
}

C.sel<-C.val[min(which.min(C.error))]
C.sel

plot(C.val,C.error,type="b")


abline(v=C.sel,lty=2)



#So we choose 0.1 as our best cost parameter. We take a look at our final SVM model on the training data
#and estimate its future performance on the test data set.

final_svm<-svm(Class~.,data=train_data,kernel="linear",cost=C.sel,type="C-classification")
summary(final_svm)


valid.pred<-predict(final_svm,valid_data)
valid.pred_percent <-  sum(diag(table(valid.pred,valid_data[,12])))/length(valid.pred)
linear.error<-1-sum(diag(table(valid.pred,valid_data[,12])))/length(valid.pred)





#We notice that the above will make the code very tedious/long when iterating throug the values of C to find the best parameter
#So the approach we will now take is using the tune to make our parameter tuning more easier and to fit models for different kernels



#Starting out with a linear kernel
linear_tune<-tune(svm, Class~.,data=train_data,type="C-classification",kernel="linear",
                  ranges=list(cost=c(0.1,0.2,1,1.5,2,3,5,7,8,9,10)))
summary(linear_tune)$best.parameters


#Looking on our accuracy for validation data and checking error for linear kernel

valid_pred_linear<-predict(linear_tune$best.model,valid_data)
linear_error<-1-sum(diag(table(valid_pred_linear,valid_data[,12])))/length(valid_pred_linear)

conf_mat_linear <- table(valid_pred_linear,valid_data$Class)
overall_class_rate_linear <- sum(diag(conf_mat_linear))/sum(conf_mat_linear)
overall_class_rate_linear



#Now lets look on the radial basis kernel

radial_tune<-tune(svm,Class~.,data=train_data,type="C-classification",kernel="radial",
                  ranges=list(cost=c(0.1,0.2,1,1.5,2,3,5,7,8,9,10)), gamma=c(0.1,0.2,0.3,0.4,0.5,1,2,3,4))

summary(radial_tune)$best.parameters


#Looking on our accuracy for validation data and checking error for radial kernel

valid_pred_radial<-predict(radial_tune$best.model,valid_data)
radial_error<- 1-sum(diag(table(valid_pred_radial,valid_data[,12])))/length(valid_pred_radial)

conf_mat_radial<- table(valid_pred_radial,valid_data$Class)
overall_class_rate_radial <- sum(diag(conf_mat_radial))/sum(conf_mat_radial)
overall_class_rate_radial



#Lastly we try the polynomial kernel with degree=2

poly_tune<-tune(svm,Class~.,data=train_data,type="C-classification",kernel="polynomial",
                degree=2,ranges=list(cost=c(0.1,1,2,3,5),gamma=c(0.01,0.02,0.05,0.1),coef0=c(0,1,2,3)))
summary(poly_tune)$best.parameters


#Try different values of the parameters

poly_tune<-tune(svm,Class~.,data=train_data,type="C-classification",kernel="polynomial",
                degree=2,ranges=list(cost=c(0.01,0.05,0.1),gamma=c(0.001,0.005,0.01),coef0=c(3,5,10)))
summary(poly_tune)$best.parameters


#Looking on our accuracy for validation data and checking error for polynomial kernel

valid_pred_poly <-predict(poly_tune$best.model,valid_data)
poly_error<-1-sum(diag(table(valid_pred_poly,valid_data[,12])))/length(valid_pred_poly)

conf_mat_poly<- table(valid_pred_poly,valid_data$Class)
overall_class_rate_poly <- sum(diag(conf_mat_poly))/sum(conf_mat_poly)
overall_class_rate_poly


#Now combining the errors and accuracies together to examine it to get an idea which kernel performed the best

all_accuracies<-matrix(c(overall_class_rate_linear,overall_class_rate_radial,overall_class_rate_poly),1,3)
colnames(all_accuracies)<-c("Linear","Radial","Polynomial (quadr.)")
round(all_accuracies,3)


all_errors<-matrix(c(linear_error,radial_error,poly_error),1,3)
colnames(all_errors)<-c("Linear","Radial","Polynomial (quadr.)")
round(all_errors,3)


#Finally we reach a conlusion in deciding wch kernel to use for our final SVM model!
#We can see that the linear kernel actually performs the best out of the three when looking on the accuracy and error rate.


#Builidng our final SVM model based on our tuning above and choosing the linear kernel

linear_tune$best.parameters  # 2

final_svm<-svm(Class~.,data=train_data,kernel="linear",cost=2,type="C-classification") #Remember cost as 2 from linear_tune variable
summary(final_svm)

pred.svm<-predict(final_svm,valid_data)
conf_mat_final_svm<- table(pred.svm,valid_data$Class)
overall_class_rate_final_svm <- sum(diag(conf_mat_final_svm))/sum(conf_mat_final_svm)
overall_class_rate_final_svm   #Final prediction on validation data


#ROC Plot for our model

# we are checking to see the performance on the validation data

#This is a function wrritten by Gareth Jameswhich was shared to us in our class  notes
rocplot<-function(pred,truth,...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)
}


fitted.valid<-attributes(predict(final_svm,valid_data,
                                 decision.values = TRUE))$decision.values


rocplot(fitted.valid,valid_data$Class,main="Validation data ROC")



#Now we will look one last final classification technique involving tree based methods

#We will start out with bagging(randomForest) and then also do a single tree 


bag_train <- randomForest(Class~.,data=train_data,mtry=6,ntree=200)
bag_train


#We can see that the out of bag estimate of error rate is 30.3%. 


predictions_bag <- predict(bag_train, valid_data,"class")
conf_mat_bag <- table(predictions_bag,valid_data$Class)
overall_class_rate_bag <- sum(diag(conf_mat_bag))/sum(conf_mat_bag)
overall_class_rate_bag


#Now lets compare our accuracy to a single tree

single_first_tree <- rpart(Class ~. ,data = train_data, method="class") #Since our dependent variable is factors 
predictions <- predict(single_first_tree, valid_data,"class")
conf_mat <- table(predictions,valid_data$Class)
overall_class_rate_single_tree <- sum(diag(conf_mat))/sum(conf_mat)
overall_class_rate_single_tree

#So we can see from above that the classification rate for a single tree is lower than that using bagging in randomForest.
#i.e Bagging performed better so this is chosen as our tree based method


#Showing Plot for the Single Tree

rpart.plot(single_first_tree,type=1,extra=4)
#Will explain further in Project Report



#Now to come to a conclusion we have used training and validation data sets for all our  classification models.

#So now we are going to use our test set on the finally selected 'best' model. Will be further explained in Project Report.

###Now let's get a quick overview on how our classifcation model performed on our validation data
#so we can choose the best model to test on our test dataset


final_accuracies<-matrix(c(pred_knn,overall_class_rate_final_svm,overall_class_rate_bag),1,3)
colnames(final_accuracies)<-c("Knn","Support Vector Machine"," Tree Based Method (Bagging)")
round(final_accuracies,3)



##Nice! so we see that the simple Knn actually turned out with a higher accuracy so we will now actually select this as our best model
##And see how well it performs on our test dataset


prediction_test<-knn(train_data[,-12], test_data[,-12], train_data[,12], k=7)  #Recall k=7 gave the best accuracy
sum((prediction_test==test_data$Class))/length(prediction_test)


# Recall that in our case  k = 7 gave the best performance on the validation data.

#We get Approximatey 79% on our testing data which is not bad but could have been better 

#Classification Rate (Based on Test data)


classification_rate <- sum((prediction_test==test_data$Class))/length(prediction_test)

##Finding Sensitivity and Specifity

cross_class_tab<-table(prediction_test,test_data[,12])
cross_class_Rates<-sweep(cross_class_tab,1,apply(cross_class_tab,1,sum),"/")

#Now getting sensitivity 

sensitivity<-cross_class_Rates[2,2]

#Lastly getting final metric, specificity

specificity<- cross_class_Rates[1,1]

#Both sensitivity and specificty seem reasonablly higer, the specifictiy would have been great if higher (we do not want drugs to be used/addicted)
#Will talk more about in the project report

### Printing metrics for best classifcation model chosen

print("Printing metrics for best classifcation model chosen")
print(classification_rate)
print(sensitivity)
print(specificity)















  