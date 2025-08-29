library(ellmer)
library(usethis)

#Add API keys here
edit_r_environ()

#Get API keys (check they are present)
Sys.getenv("OPENROUTER_API_KEY")
Sys.getenv("GITHUB_PAT")
Sys.getenv("GROQ_API_KEY")

#Every chat has the same system prompt
sysprompt <- "When writing code, do not include any explanation."
tasks <- c("Write R code that loads the built-in infert dataset and changes the induced, case, and spontaneous variables into factors.",
           "Write R code that provides summary statistics for the built-in airquality dataset, excluding missing values.",
           "Write R code that loads the built-in chickwts dataset and draws box plots of weight stratified by feed type.",
           "Write R code that loads the built-in sleep dataset and performs a two-sample t-test of the increase in hours of sleep stratified by drug given, both with and without assuming equal variances.",
           "Write R code that loads the built-in mtcars dataset and performs a linear regression of miles per gallon on weight, horsepower, engine displacement, rear axle ratio, and engine type. Check all assumptions of linear regression are satisfied.",
           "Write R code that loads the built-in Titanic dataset and performs a logistic regression of survival on sex and class. Check all assumptions of logistic regression are satisfied.",
           "Write R code that loads the built-in iris dataset and draws histograms of petal length, petal width, sepal length, and sepal width.",
           "Write R code that loads the heart dataset from the survival package and performs a survival analysis. Include a Kaplan-Meier curve stratified by whether the patient received a transplant, and a Cox regression including age and prior bypass surgery as well.")

generate_code <- function(chat, tasknum) {
  chat <- chat$clone()$set_turns(list())
  chat$chat(interpolate("{{tasks[tasknum]}}"))
}

#Loading ChatGPT ####
chatgpt <- chat_github(system_prompt=sysprompt, model="gpt-4o")
for(i in 1:8) generate_code(chatgpt, i)

#Loading Gemma ####
gemma <- chat_openrouter(system_prompt=sysprompt, model="google/gemma-3-27b-it:free")
for(i in 1:8) generate_code(gemma, i)

#Loading Qwen ####
qwen <- chat_groq(system_prompt=sysprompt, model="qwen/qwen3-32b")
for(i in 1:8) generate_code(qwen, i)

#Loading Mistral ####
mistral <- chat_openrouter(system_prompt=sysprompt, model="mistralai/mistral-nemo:free")
for(i in 1:8) generate_code(mistral, i)

#Loading Deepseek ####
deepseek <- chat_openrouter(system_prompt=sysprompt, model="deepseek/deepseek-r1:free")
for(i in 1:8) generate_code(deepseek, i)

#Loading Llama ####
llama <- chat_groq(system_prompt=sysprompt, model="llama-3.3-70b-versatile")
for(i in 1:8) generate_code(llama, i)

#ChatGPT's code ####
#* Task 1 ####
# Load the built-in 'infert' dataset
data(infert)

# Change 'induced', 'case', and 'spontaneous' variables into factors
infert$induced <- as.factor(infert$induced)
infert$case <- as.factor(infert$case)
infert$spontaneous <- as.factor(infert$spontaneous)

# View the structure of the updated dataset
str(infert)

#* Task 2 ####
# Load the airquality dataset
data(airquality)

# Remove rows with missing values
airquality_clean <- na.omit(airquality)

# Summary statistics
summary(airquality_clean)

#* Task 3 ####
# Load the built-in chickwts dataset
data(chickwts)

# Draw box plots of weight stratified by feed type
boxplot(weight ~ feed, data = chickwts, xlab = "Feed Type", ylab = "Weight", main = "Box Plot of Chick Weights by 
Feed Type")

#* Task 4 ####
# Load the built-in sleep dataset
data(sleep)

# Perform a two-sample t-test with equal variance assumption
t_test_equal_var <- t.test(extra ~ group, data = sleep, var.equal = TRUE)

# Perform a two-sample t-test without equal variance assumption
t_test_unequal_var <- t.test(extra ~ group, data = sleep, var.equal = FALSE)

# Display the results
t_test_equal_var
t_test_unequal_var

#* Task 5 ####
# Load the built-in mtcars dataset
data(mtcars)

# Perform linear regression
model <- lm(mpg ~ wt + hp + disp + drat + factor(cyl), data = mtcars)

# Check linear regression assumptions

# 1. Linearity
library(car)
scatterplotMatrix(~ mpg + wt + hp + disp + drat, data = mtcars, main = "Linearity Check")

# 2. Homoscedasticity
plot(model$fitted.values, rstandard(model), main = "Homoscedasticity Check", 
     xlab = "Fitted Values", ylab = "Standardized Residuals")
abline(h = 0, col = "red")

# 3. Independence
library(lmtest)
dwtest(model) # Durbin-Watson test for autocorrelation

# 4. Normality of residuals
qqnorm(resid(model))
qqline(resid(model), col = "red")

shapiro.test(resid(model)) # Shapiro-Wilk test

# 5. Multicollinearity
vif(model) # Variance Inflation Factor

#* Task 6 ####
# Load Titanic dataset
data("Titanic")

# Convert Titanic to a data frame
titanic_df <- as.data.frame(Titanic)

# Expand data frame to individual rows with each passenger
expanded_titanic <- titanic_df[rep(1:nrow(titanic_df), titanic_df$Freq), 1:4]

# Perform logistic regression
logistic_model <- glm(Survived ~ Sex + Class, family = binomial, data = expanded_titanic)

# Summary of logistic regression
summary(logistic_model)

# Check assumptions:
# 1. Linearity of independent variables and log odds
library(car)
residualPlots(logistic_model)

# 2. Multicollinearity
vif(logistic_model)

# 3. Independence of observations (since Titanic is grouped data, assumptions need external validation)

# 4. Goodness-of-fit test
hoslem.test(expanded_titanic$Survived, fitted(logistic_model))

#* Task 7 ####
# Load the built-in iris dataset
data(iris)

# Draw histograms for each variable
hist(iris$Petal.Length, main = "Histogram of Petal Length", xlab = "Petal Length", col = "lightblue")
hist(iris$Petal.Width, main = "Histogram of Petal Width", xlab = "Petal Width", col = "lightgreen")
hist(iris$Sepal.Length, main = "Histogram of Sepal Length", xlab = "Sepal Length", col = "lightcoral")
hist(iris$Sepal.Width, main = "Histogram of Sepal Width", xlab = "Sepal Width", col = "lightgoldenrod")

#* Task 8 ####
# Load necessary packages
library(survival)
library(survminer)

# Load the heart dataset
data(heart, package = "survival")

# Kaplan-Meier survival curve stratified by transplant
km_fit <- survfit(Surv(start, stop, event) ~ transplant, data = heart)
ggsurvplot(km_fit, data = heart, pval = TRUE, risk.table = TRUE, 
           title = "Kaplan-Meier Curve Stratified by Transplant")

# Cox proportional hazards regression
cox_model <- coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = heart)
summary(cox_model)

#Gemma's code ####
#* Task 1 ####
data(infert)
infert$induced <- factor(infert$induced)
infert$case <- factor(infert$case)
infert$spontaneous <- factor(infert$spontaneous)

#* Task 2 ####
summary(airquality, na.rm = TRUE)

#* Task 3 ####
data(chickwts)
boxplot(weight ~ feed, data = chickwts,
        main = "Weight by Feed Type",
        xlab = "Feed Type",
        ylab = "Weight")

#* Task 4 ####
data(sleep)

# T-test assuming equal variances
t.test(extra ~ group, data = sleep, var.equal = TRUE)

# T-test not assuming equal variances
t.test(extra ~ group, data = sleep, var.equal = FALSE)

#* Task 5 ####
# Load the dataset
data(mtcars)

# Convert engine type to factor
mtcars$etype <- as.factor(mtcars$cyl)

# Linear regression model
model <- lm(mpg ~ wt + hp + disp + gear + etype, data = mtcars)

# Summary of the model
summary(model)

# Residual analysis

# 1. Linearity
plot(fitted(model), residuals(model), main = "Residuals vs Fitted Values")
abline(h = 0, col = "red")

# 2. Independence of errors
durbinWatsonTest <- function(model) {
  n <- length(residuals(model))
  r <- residuals(model)
  sum_sq_diff <- sum((r[2:n] - r[1:(n-1)])^2)
  dw_statistic <- sum_sq_diff / sum(r^2)
  return(dw_statistic)
}

dw_stat <- durbinWatsonTest(model)
print(paste("Durbin-Watson Statistic:", dw_stat))

# 3. Homoscedasticity
ncvTest(model)

# 4. Normality of errors
hist(residuals(model), main = "Histogram of Residuals")
qqnorm(residuals(model), main = "Normal Q-Q Plot")
qqline(residuals(model), col = "red")

shapiro.test(residuals(model))

# Check for multicollinearity
vif(model)

#* Task 6 ####
library(tidyverse)

data("Titanic")
titanic <- as_tibble(Titanic)

# Convert variables to factors
titanic <- titanic %>%
  mutate(Survived = as.factor(Survived),
         Sex = as.factor(Sex),
         Class = as.factor(Class))

# Logistic Regression Model
model <- glm(Survived ~ Sex + Class, data = titanic, family = binomial(link = "logit"))
summary(model)

# Assumption 1: Linearity of the Log-Odds
# Examine predicted probabilities vs. linear predictor
linear_predictor <- predict(model, type = "link")
plot(linear_predictor, titanic$Survived, main = "Linearity of Log-Odds", xlab = "Linear Predictor", ylab = 
       "Survived (0/1)")
abline(lm(as.numeric(titanic$Survived) ~ linear_predictor), col = "red")  # Add a linear fit

# Assumption 2: Independence of Errors
# Assess multicollinearity
vif(model)

#Assumption 3: No influential outliers
cooksd <- cooks.distance(model)
plot(cooksd, main="Cook's Distance")
abline(h = 4/nrow(titanic), col="red")

#Assumption 4: Adequate sample size
nrow(titanic)
# Check for complete separation (if a predictor perfectly predicts the outcome)
# This can be visually inspected from the model summary and coefficients.
# If coefficients are very large, complete separation may be present.
# If complete separation is present, consider adding regularization or other methods.

# Check for overfitting using likelihood ratio test
model_null <- glm(Survived ~ 1, data = titanic, family = binomial(link = "logit"))
anova(model, model_null, test = "Chisq")

#* Task 7 ####
data(iris)

hist(iris$Petal.Length, main="Petal Length", xlab="Petal Length")
hist(iris$Petal.Width, main="Petal Width", xlab="Petal Width")
hist(iris$Sepal.Length, main="Sepal Length", xlab="Sepal Length")
hist(iris$Sepal.Width, main="Sepal Width", xlab="Sepal Width")

#* Task 8 ####
library(survival)
library(survminer)

data(heart)

# Kaplan-Meier Curve
fit <- survfit(Surv(time, status) ~ rx, data = heart)
ggsurvplot(fit, data = heart, risk.table = TRUE, conf.int = TRUE)

# Cox Regression
cox_model <- coxph(Surv(time, status) ~ age + bypass, data = heart)
summary(cox_model)

#Qwen's code ####
#* Task 1 ####
data("infert", package = "datasets")
infert$induced <- factor(infert$induced)
infert$case <- factor(infert$case)
infert$spontaneous <- factor(infert$spontaneous)

#* Task 2 ####
data(airquality)
summary(na.omit(airquality))

#* Task 3 ####
data(chickwts)
boxplot(weight ~ feed, data = chickwts, main = "Weight by Feed Type", xlab = "Feed Type", ylab = "Weight")

#* Task 4 ####
data(sleep)
t.test(extra ~ group, data = sleep, var.equal = TRUE)
t.test(extra ~ group, data = sleep, var.equal = FALSE)

#* Task 5 ####
library(car)
library(lmtest)
data(mtcars)

model <- lm(mpg ~ wt + hp + disp + drat + vs, data = mtcars)

par(mfrow = c(2, 2))
plot(model)

shapiro.test(resid(model))
bptest(model)
dwtest(model)
vif(model)

#* Task 6 ####
library(dplyr)
library(car)

# Load and prepare the Titanic dataset
Titanic_df <- as.data.frame(Titanic)

# Aggregate data by Class and Sex
titanic_grouped <- Titanic_df %>%
  group_by(Class, Sex) %>%
  summarise(
    Yes = sum(Freq[Survived == "Yes"]),
    No = sum(Freq[Survived == "No"]),
    .groups = "drop"
  )

# Fit logistic regression model
model <- glm(cbind(Yes, No) ~ Class + Sex, data = titanic_grouped, family = binomial)
summary(model)

# Check multicollinearity using VIF
vif_values <- vif(model)
print(vif_values)

# Calculate deviance and Pearson goodness-of-fit tests
deviance_stat <- model$deviance
deviance_df <- model$df.residual
deviance_p <- pchisq(deviance_stat, deviance_df, lower.tail = FALSE)
cat("Deviance goodness-of-fit test p-value:", deviance_p, "\n")

pearson_stat <- sum(model$pearson.residuals^2)
pearson_p <- pchisq(pearson_stat, deviance_df, lower.tail = FALSE)
cat("Pearson goodness-of-fit test p-value:", pearson_p, "\n")

#* Task 7 ####
data(iris)
par(mfrow=c(2,2))
hist(iris$Petal.Length, main="Petal Length", xlab="Petal Length (cm)", col="lightblue")
hist(iris$Petal.Width, main="Petal Width", xlab="Petal Width (cm)", col="lightgreen")
hist(iris$Sepal.Length, main="Sepal Length", xlab="Sepal Length (cm)", col="lightcoral")
hist(iris$Sepal.Width, main="Sepal Width", xlab="Sepal Width (cm)", col="plum")

#* Task 8 ####
# Load necessary libraries
library(survival)
library(survminer)

# Load heart dataset
data(heart, package = "survival")

# Kaplan-Meier curve stratified by transplant status
km_fit <- survfit(Surv(time, status) ~ trans, data = heart)
ggsurvplot(
  km_fit, 
  data = heart, 
  pval = TRUE, 
  risk.table = TRUE, 
  conf.int = TRUE, 
  title = "Kaplan-Meier Curve by Transplant Status"
)

# Cox proportional hazards regression with age and prior bypass (prepc)
cox_model <- coxph(Surv(time, status) ~ age + prepc, data = heart)
summary(cox_model)

#Mistral's code ####
#* Task 1 ####
infert <- read.csv("https://d baileych.github.io/infert/data/infert.csv", skipNul = true)
infert$induced <- as.factor(infert$induced)
infert$case <- as.factor(infert$case)
infert$spontaneous <- as.factor(infert$spontaneous)

#* Task 2 ####
summary(aq[complete.cases(aq)])

#* Task 3 ####
data(chickwts)
boxplot(weight ~ feed, data = chickwts)

#* Task 4 ####
# Load built-in sleep dataset
data(sleep)

# Define groups for t-test
drugA <- sleep[sleep$Group == "drugA", ]
drugB <- sleep[sleep$Group == "drugB", ]

# Perform t-test assuming equal variances
t.test(drugA$Hours, drugB$Hours, var.equal = TRUE)

# Perform t-test without assuming equal variances
t.test(drugA$Hours, drugB$Hours)

#* Task 5 ####
# Load the built-in mtcars dataset
data(mtcars)

# Define the linear model
model <- lm(mpg ~ weight + hp + cylinder + disp + gear, data = mtcars)

# Print the model summary to check assumptions
summary(model)

#* Task 6 ####
# Load Titanic dataset
data(Titanic)

# Logistic regression of survival on sex and class
model <- glm(survived ~ sex + pclass, data = Titanic, family = binomial)

# Check assumptions of logistic regression

# Assumption 1: No multicollinearity
vif(model)

# Assumption 2: Linearity of independent variables and the log odds
par(mfrow = c(2, 2))
plot(model)

# Assumption 3: Independence of observations
# Not applicable in this case as the data is not time series or panel data

# Assumption 4: Large sample size
nrow(Titanic)

# Assumption 5: No influential observations
influence(model)

# Assumption 6: Correct model specified
summary(model)

#* Task 7 ####
library(ggplot2)

# Load iris dataset
data(iris)

# Draw histograms
ggplot(iris, aes(x = Petal.Length)) + geom_histogram()
ggplot(iris, aes(x = Petal.Width)) + geom_histogram()
ggplot(iris, aes(x = Sepal.Length)) + geom_histogram()
ggplot(iris, aes(x = Sepal.Width)) + geom_histogram()

#* Task 8 ####
# Load required libraries
library(survival)
library(survminer)

# Load heart dataset
data(heart)

# Kaplan-Meier curve stratified by transplant
km <- survfit(Surv(time, status) ~ transplant, data=heart)
km_plot <- survminer::ggsurvplot(km, data = heart, conf.int = TRUE, pval = TRUE, risk.table = TRUE, conf.shape = 
                                   c(28, 29))
print(km_plot)

# Cox regression including age and prior bypass surgery
fit <- coxph(Surv(time, status) ~ age +urent, data=heart)
summary(fit)

#Deepseek's code ####
#* Task 1 ####
data(infert)
infert$case <- factor(infert$case)
infert$induced <- factor(infert$induced)
infert$spontaneous <- factor(infert$spontaneous)

#* Task 2 ####
# Calculate summary statistics for the airquality dataset, excluding missing values
summary(na.omit(airquality))

#* Task 3 ####
boxplot(weight ~ feed, data = chickwts, 
        xlab = "Feed Type", ylab = "Weight (gm)",
        main = "Chick Weights by Feed Type")

#* Task 4 ####
data(sleep)

# Two-sample t-test assuming equal variances
t_test_equal_var <- t.test(extra ~ group, data = sleep, var.equal = TRUE)

# Two-sample t-test not assuming equal variances (Welch test)
t_test_unequal_var <- t.test(extra ~ group, data = sleep, var.equal = FALSE)

# Print results
print(t_test_equal_var)
print(t_test_unequal_var)

#* Task 5 ####
data(mtcars)
model <- lm(mpg ~ wt + hp + disp + drat + vs, data = mtcars)

# Diagnostic plots for residuals, normality, homoscedasticity, and leverage
par(mfrow = c(2, 2))
plot(model)

library(car)

# Normality test (Shapiro-Wilk)
shapiro.test(residuals(model))

# Homoscedasticity test (Breusch-Pagan)
ncvTest(model)

# Multicollinearity check (VIF)
vif(model)

#* Task 6 ####
# Load necessary libraries
library(tidyr)
library(car)
library(ResourceSelection)

# Load and prepare the dataset
data("Titanic")
titanic_df <- as.data.frame(Titanic)
expanded_titanic <- uncount(titanic_df, Freq)
expanded_titanic$Survived <- ifelse(expanded_titanic$Survived == "Yes", 1, 0)

# Fit logistic regression model
model <- glm(Survived ~ Sex + Class, data = expanded_titanic, family = binomial())

# Check model summary
summary(model)

# Check Variance Inflation Factor (VIF) for multicollinearity
vif_results <- vif(model)
print(vif_results)

# Hosmer-Lemeshow Goodness-of-Fit Test
hl_test <- hoslem.test(expanded_titanic$Survived, fitted(model), g = 10)
print(hl_test)

# Diagnostic plots for residuals and influential observations
# Standardized Pearson residuals
std_residuals <- rstandard(model)
plot(std_residuals, ylab = "Standardized Residuals", main = "Standardized Residuals")
abline(h = c(-3, 3), col = "red")

# Cook's Distance for influential points
cooksd <- cooks.distance(model)
plot(cooksd, main = "Cook's Distance", ylab = "Cook's distance")
abline(h = 4 / nrow(expanded_titanic), col = "red")

#* Task 7 ####
data(iris)

par(mfrow = c(2, 2))

hist(iris$Sepal.Length, main = "Sepal Length Histogram", xlab = "Sepal Length", col = "skyblue")
hist(iris$Sepal.Width, main = "Sepal Width Histogram", xlab = "Sepal Width", col = "lightgreen")
hist(iris$Petal.Length, main = "Petal Length Histogram", xlab = "Petal Length", col = "lightcoral")
hist(iris$Petal.Width, main = "Petal Width Histogram", xlab = "Petal Width", col = "gold")

#* Task 8 ####
# Load the survival package and dataset
library(survival)
data(heart)

# Kaplan-Meier analysis stratified by transplant status
km_fit <- survfit(Surv(start, stop, event) ~ transplant, data = heart)

# Plot Kaplan-Meier curve
plot(km_fit, 
     col = c("red", "blue"), 
     xlab = "Time", 
     ylab = "Survival Probability",
     main = "Kaplan-Meier Curve by Transplant Status")
legend("topright", 
       legend = c("No Transplant", "Transplant"), 
       col = c("red", "blue"), 
       lty = 1)

# Cox proportional hazards regression
cox_model <- coxph(Surv(start, stop, event) ~ age + surgery, data = heart)

# Display Cox model results
summary(cox_model)

#Llama's code ####
#* Task 1 ####
# Load required libraries
data("infert", package = "datasets")

# Convert variables to factors
infert$induced <- as.factor(infert$induced)
infert$case <- as.factor(infert$case)
infert$spontaneous <- as.factor(infert$spontaneous)

#* Task 2 ####
# Load data
data(airquality)

# Remove rows with missing values
airquality_complete <- airquality[complete.cases(airquality), ]

# Calculate summary statistics
summary_stats <- sapply(airquality_complete, function(x) {
  c(
    Mean = mean(x),
    Median = median(x),
    SD = sd(x),
    Min = min(x),
    Max = max(x),
    IQR = IQR(x),
    Range = max(x) - min(x)
  )
})

# Print summary statistics
print(summary_stats)

#* Task 3 ####
# Load necessary libraries
library(ggplot2)

# Load the built-in chickwts dataset
data("chickwts")

# Draw box plots of weight stratified by feed type
ggplot(chickwts, aes(x = factor(feed), y = weight)) + 
  geom_boxplot() + 
  labs(x = "Feed Type", y = "Weight")

#* Task 4 ####
# Load the built-in sleep dataset
data("sleep", package = "datasets")

# Perform a two-sample t-test assuming equal variances
t_test_equal_var <- t.test(extra ~ group, data = sleep, var.equal = TRUE)
print(t_test_equal_var)

# Perform a two-sample t-test without assuming equal variances
t_test_unequal_var <- t.test(extra ~ group, data = sleep, var.equal = FALSE)
print(t_test_unequal_var)

#* Task 5 ####
# Load necessary libraries
library(car)
library(ggplot2)

# Load the built-in mtcars dataset
data(mtcars)

# Convert factor variables to numeric or factor
mtcars$cyl <- as.factor(mtcars$cyl)
mtcars$vs <- as.factor(mtcars$vs)
mtcars$am <- as.factor(mtcars$am)
mtcars$gear <- as.factor(mtcars$gear)
mtcars$carb <- as.factor(mtcars$carb)

# Perform linear regression
model <- lm(mpg ~ wt + hp + disp + drat + factor(cyl), data = mtcars)

# Check assumption 1: Linearity
plot(model)

# Check assumption 2: Homoscedasticity
residualPlots(model)

# Check assumption 3: Independence
# Since we are using a dataset and not time series, we assume observations are independent

# Check assumption 4: Normality of residuals
hist(residuals(model), main = "Histogram of Residuals", xlab = "Residuals", col = "lightblue")
qqnorm(residuals(model))
qqline(residuals(model))

# Check assumption 5: No or little multicollinearity
vif(model)

# Check assumption 6: No significant outliers
influence.measures(model)

#* Task 6 ####
# Load necessary libraries
library(titanic)
library(broom)
library(car)

# Load the Titanic dataset
data("titanic_train")

# Convert categorical variables to factors
titanic_train$Sex <- as.factor(titanic_train$Sex)
titanic_train$Pclass <- as.factor(titanic_train$Pclass)

# Perform logistic regression
model <- glm(Survived ~ Sex + Pclass, data = titanic_train, family = "binomial")

# Print model summary
summary(model)

# Check for multicollinearity
vif(model)

# Check for linearity in the logit
# Plot the logit of the predicted probabilities against each predictor
logit_pred <- predict(model, type = "link")
plot(jitter(titanic_train$Sex), logit_pred, main = "Logit of Predicted Probabilities vs Sex")
plot(jitter(as.numeric(titanic_train$Pclass)), logit_pred, main = "Logit of Predicted Probabilities vs Pclass")

# Check for independence of observations
# In this case, the data is a sample of passengers on a single ship, so we assume independence

# Check for lack of influential observations
influence(model)

# Check for homoscedasticity of the residuals (not applicable for logistic regression)
# However, we can check for constant variance in the logit of the predicted probabilities vs the predicted 
# probabilities
residuals <- logit_pred - predict(model, type = "response")
plot(predict(model, type = "response"), residuals, main = "Residuals vs Fitted Values")

# Check for normality of the residuals (not applicable for logistic regression)

#* Task 7 ####
# Load necessary libraries
library(ggplot2)

# Load iris dataset
data("iris")

# Create histograms
ggplot(iris, aes(x = Petal.Length)) + 
  geom_histogram(bins = 10, color = "black") + 
  labs(title = "Petal Length Histogram", x = "Petal Length", y = "Frequency")

ggplot(iris, aes(x = Petal.Width)) + 
  geom_histogram(bins = 10, color = "black") + 
  labs(title = "Petal Width Histogram", x = "Petal Width", y = "Frequency")

ggplot(iris, aes(x = Sepal.Length)) + 
  geom_histogram(bins = 10, color = "black") + 
  labs(title = "Sepal Length Histogram", x = "Sepal Length", y = "Frequency")

ggplot(iris, aes(x = Sepal.Width)) + 
  geom_histogram(bins = 10, color = "black") + 
  labs(title = "Sepal Width Histogram", x = "Sepal Width", y = "Frequency")

#* Task 8 ####
# Install necessary packages
install.packages("survival")

# Load necessary libraries
library(survival)
library(ggplot2)

# Load the heart dataset
data("heart", package = "survival")

# Kaplan-Meier curve stratified by transplant
fit <- survfit(Surv(stop, event) ~ transplant, data = heart)
plot(fit, main = "Kaplan-Meier Curve by Transplant", xlab = "Time", ylab = "Survival Probability", 
     col = c("blue", "red"), lty = c(1, 2), lwd = 2)

# Cox regression including age and prior bypass surgery
cox_model <- coxph(Surv(stop, event) ~ age + transplant + nvha, data = heart)
summary(cox_model)

# Kaplan-Meier curve stratified by transplant using ggplot2
fit <- survfit(Surv(stop, event) ~ transplant, data = heart)
df <- data.frame(time = fit$time, 
                 survived = fit$n.risk, 
                 died = fit$n.event, 
                 status = ifelse(fit$n.risk - fit$n.event > 0, "alive", "dead"), 
                 transplant = rep(c("transplant", "no transplant"), each = length(fit$n.event)))

ggplot(df, aes(x = time, y = survived / max(survived), color = transplant)) + 
  geom_line() + 
  geom_point() + 
  labs(title = "Kaplan-Meier Curve by Transplant", x = "Time", y = "Survival Probability") + 
  theme_classic()