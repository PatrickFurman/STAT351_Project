# Load libraries
library(ggplot2)
library(rpart)
library(rpart.plot)
library(catboost)
library(randomForest)
library(Metrics)
library(varhandle)
library(performanceEstimation)

# Import data
df = read.csv('heart_2020_cleaned.csv')
#View(df)
str(df)

# Fix data types
df[df$HeartDisease == 'Yes','HeartDisease'] = 1
df[df$HeartDisease == 'No','HeartDisease'] = 0
numeric_col_indices = c(2, 6, 7, 15)
df[,-numeric_col_indices] = lapply(df[,-numeric_col_indices], as.factor)

# Train/test split method
split_data = function(data, seed=1234, train_perc=0.7) {
  set.seed(seed)
  indices = sample(1:nrow(data), nrow(data)*train_perc, replace=F)
  tr = data[indices,]
  te = data[-indices,]
  return(list(tr, te))
}

# Original data split
x = split_data(df)
train=x[[1]]
test=x[[2]]

# Undersampling
# df2 is a new dataframe with all the positive heart disease cases and an equal
# number of randomy selected negative cases
hd_yes = which(df$HeartDisease == '1')
df_no = df[-hd_yes,]
selected_no = sample(1:nrow(df_no), length(hd_yes), replace=F)
df_no = df_no[selected_no,]
df2 = rbind(df_no, df[hd_yes,])
x = split_data(df2)
train2=x[[1]]
test2=x[[2]]

# Oversampling using SMOTE
df3 = smote(HeartDisease~., df, perc.over=2, k=5, perc.under=2)
x = split_data(df3)
train3=x[[1]]
test3=x[[2]]
df4 = smote(HeartDisease~., df, perc.over=3, k=5, perc.under=2)
x = split_data(df4)
train4=x[[1]]
test4=x[[2]]
df5 = smote(HeartDisease~., df, perc.over=4, k=5, perc.under=2)
x = split_data(df5)
train5=x[[1]]
test5=x[[2]]

# Dataframe of all datasets
datasets = list('original'=list('tr'=train, 'te'=test),
                'undersampled'=list('tr'=train2, 'te'=test2),
                'oversampled_2'=list('tr'=train3, 'te'=test3),
                'oversampled_3'=list('tr'=train4, 'te'=test4),
                'oversampled_4'=list('tr'=train5, 'te'=test5))

# Evaluation method for models below
evaluate = function(model, tr, te, model_name, data_name=NA) {
  # Create predictions
  train_preds = predict(model, tr, type='class')
  test_preds = predict(model, te, type='class')
  
  # Evaluate predictions and return results
  res = data.frame(model=model_name,
                   data=data_name,
                   train_accuracy=accuracy(tr$HeartDisease, train_preds),
                   train_f1=fbeta_score(unfactor(tr$HeartDisease), unfactor(train_preds)),
                   train_precision=precision(unfactor(tr$HeartDisease), unfactor(train_preds)),
                   train_recall=recall(unfactor(tr$HeartDisease), unfactor(train_preds)),
                   test_accuracy=accuracy(te$HeartDisease, test_preds),
                   test_f1=fbeta_score(unfactor(te$HeartDisease), unfactor(test_preds)),
                   test_precision=precision(unfactor(te$HeartDisease), unfactor(test_preds)),
                   test_recall=recall(unfactor(te$HeartDisease), unfactor(test_preds)))
  return(res)
}

# Dataframe to hold results for testing all models all all datasets
comparison = data.frame(model=NA, data=NA, train_accuracy=NA, train_f1=NA,
                        train_precision=NA, train_recall=NA, test_accuracy=NA,
                        test_f1=NA, test_precision=NA, test_recall=NA)

# Simple CART models
for (i in 1:length(datasets)) {
  tr = datasets[[i]][[1]]
  te = datasets[[i]][[2]]
  tree = rpart(HeartDisease~., data=tr)
  comparison = rbind(comparison, evaluate(tree, tr, te, 'basic tree', names(datasets)[i]))
}

# Random Forest models
for (i in 1:length(datasets)) {
  tr = datasets[[i]][[1]]
  te = datasets[[i]][[2]]
  rf = randomForest(HeartDisease~., data=tr)
  comparison = rbind(comparison, evaluate(rf, tr, te, 'rf', names(datasets)[i]))
}
# rf = rpart(HeartDisease~., data=tr, ntree=500, cutoff=c(0.55, 0.45), maxnodes=5)

# Catboost models
for (i in 1:length(datasets)) {
  tr = datasets[[i]][[1]]
  te = datasets[[i]][[2]]
  tr_features = tr[,-c(1)]
  tr_labels = unfactor(tr$HeartDisease)
  train_pool = catboost.load_pool(data=tr_features, label=tr_labels)
  te_features = te[,-c(1)]
  te_labels = unfactor(te$HeartDisease)
  test_pool = catboost.load_pool(data=te_features, label=te_labels)
  cat_model = catboost.train(train_pool, NULL, params=list(loss_function='CrossEntropy',
                                                           iterations=1000))
  train_preds = catboost.predict(cat_model, train_pool, prediction_type='Class')
  test_preds = catboost.predict(cat_model, test_pool, prediction_type='Class')
  row = data.frame(model='catboost',
                   data=names(datasets)[i],
                   train_accuracy=accuracy(tr$HeartDisease, train_preds),
                   train_f1=fbeta_score(unfactor(tr$HeartDisease), train_preds),
                   train_precision=precision(unfactor(tr$HeartDisease), train_preds),
                   train_recall=recall(unfactor(tr$HeartDisease), train_preds),
                   test_accuracy=accuracy(te$HeartDisease, test_preds),
                   test_f1=fbeta_score(unfactor(te$HeartDisease), test_preds),
                   test_precision=precision(unfactor(te$HeartDisease), test_preds),
                   test_recall=recall(unfactor(te$HeartDisease), test_preds))
  comparison = rbind(comparison, row)
}

# Remove initial row of NAs from comparison dataframe
comparison = comparison[-c(1),]

# Grid search for hyperparameter tuning
depths = c(2, 6, 8)
ntrees = c(150, 500)

results = data.frame(depth=rep(depths, length(ntrees)), 
                        ntrees=unlist(lapply(ntrees, function(x){return(rep(x, length(depths)))})),
                        train_accuracy=rep(NA, length(depths)*length(ntrees)),
                        train_f1=rep(NA, length(depths)*length(ntrees)),
                        train_precision=rep(NA, length(depths)*length(ntrees)),
                        train_recall=rep(NA, length(depths)*length(ntrees)),
                        test_accuracy=rep(NA, length(depths)*length(ntrees)),
                        test_f1=rep(NA, length(depths)*length(ntrees)),
                        test_precision=rep(NA, length(depths)*length(ntrees)),
                        test_recall=rep(NA, length(depths)*length(ntrees)))
for (n in ntrees) {
  for (depth in depths) {
    rf = randomForest(HeartDisease~., data=train5, ntree=n, maxnodes=depth)
    row = evaluate(rf, train5, test5, 'rf_tune')
    results[results$depth == depth & results$ntrees == n,-c(1,2)] = row[,-c(1,2)]
  }
}

results
best_n = NA
best_depth = NA
#write.csv(results, 'tuning_results.csv', row.names=F)

# 5 fold cross validation for evaluating final model
k = 5
cutoffs = round(seq(1, nrow(df5), nrow(df5)/k))
accs = rep(NA, k)
f1s = rep(NA, k)
for (i in 1:(length(cutoffs)-1)) {
  start = cutoffs[i]
  end = cutoffs[i+1]
  indices = start:end
  test_cv = df5[indices,]
  train_cv = df5[-indices,]
  rf_cv = randomForest(HeartDisease~., data=train_cv)
  cv_preds = predict(rf_cv, test_cv, type='Class')
  accs[i] = accuracy(test_cv$HeartDisease, cv_preds)
  f1s[i] = fbeta_score(unfactor(test_cv$HeartDisease), unfactor(cv_preds))
}
indices = cutoffs[5]:nrow(df5)
test_cv = df5[indices,]
train_cv = df5[-indices,]
rf_cv = randomForest(HeartDisease~., data=train_cv)
cv_preds = predict(rf_cv, test_cv, type='Class')
accs[5] = accuracy(test_cv$HeartDisease, cv_preds)
f1s[5] = fbeta_score(unfactor(test_cv$HeartDisease), unfactor(cv_preds))

cat('Final model test F1 score from 5 fold cross validation:', mean(f1s, na.rm=T))
cat('Final model test accuracy from 5 fold cross validation:', mean(accs, na.rm=T))

