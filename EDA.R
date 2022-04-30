# Load libraries
library(ggplot2)

# Import data
df = read.csv('heart_2020_cleaned.csv')
View(df)
str(df)

# Fix data types
numeric_col_indices = c(2, 6, 7, 15)
df[,-numeric_col_indices] = lapply(df[,-numeric_col_indices], as.factor)

# Number of unique elements for each category
num_unique = data.frame(Num_Unique = apply(df, 2, function(x) {return(length(unique(x)))}))
num_unique

# Checking for missing values
num_na = data.frame(Num_NA = apply(df, 2, function(x) {return(sum(is.na(x)))}))
num_na

# Violin plots for numerical features
ggplot(data=df, aes(x=HeartDisease, y=BMI, col=HeartDisease)) + 
  geom_violin(draw_quantiles = c(.25, .50, .75))
ggplot(data=df, aes(x=HeartDisease, y=PhysicalHealth, col=HeartDisease)) + 
  geom_violin(draw_quantiles = c(.25, .50, .75))
ggplot(data=df, aes(x=HeartDisease, y=MentalHealth, col=HeartDisease)) + 
  geom_violin(draw_quantiles = c(.25, .50, .75))
ggplot(data=df, aes(x=HeartDisease, y=SleepTime, col=HeartDisease)) + 
  geom_violin(draw_quantiles = c(.25, .50, .75))

# Mosaic plots for categorical features
mosaicplot(df$HeartDisease~df$Smoking)
mosaicplot(df$HeartDisease~df$AlcoholDrinking)
mosaicplot(df$HeartDisease~df$Stroke)
mosaicplot(df$HeartDisease~df$DiffWalking)
mosaicplot(df$HeartDisease~df$Sex)
mosaicplot(df$HeartDisease~df$PhysicalActivity)
mosaicplot(df$HeartDisease~df$Asthma)
mosaicplot(df$HeartDisease~df$KidneyDisease)
mosaicplot(df$HeartDisease~df$SkinCancer)

# Stacked bar charts for categorical variables with more than two categories
ggplot(data=df, aes(x=Diabetic, fill=HeartDisease)) + 
  geom_bar(position='fill') + 
  labs(title='Proportion of People with Heart Disease in Each Diabetic Category',
       y='Proportion')
ggplot(data=df, aes(x=GenHealth, fill=HeartDisease)) + 
  geom_bar(position='fill') + 
  labs(title='Proportion of People with Heart Disease in Each GenHealth Category',
       y='Proportion')
ggplot(data=df, aes(x=Race, fill=HeartDisease)) + 
  geom_bar(position='fill') + 
  labs(title='Proportion of People with Heart Disease by Race',
       y='Proportion')
ggplot(data=df, aes(x=AgeCategory, fill=HeartDisease)) + 
  geom_bar(position='fill') + 
  labs(title='Proportion of People with Heart Disease in Each Age Category',
       y='Proportion')
