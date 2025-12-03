
###############################################
# SVM Application: Breast Cancer Classification
# Research Question:
# "Is Support Vector Machines an accurate method that predicts
#  whether a cell is malignant or benign?"
###############################################

###########
# Libraries
###########
library(e1071)      # SVM (linear & radial)
library(ggplot2)    # Plots
library(caret)      # Confusion matrix + tuning
library(kknn)
library(knitr)      # Tables
library(kableExtra) # Pretty tables

#####################
# Data Preparation
#####################

# Set working directory
setwd("/Users/veragakanagrova/Desktop/Erasmus Rotterdam University/Classes/Data")

# Load data
df <- read.csv("breast_cancer.csv")

# Basic checks
str(df)
summary(df)
colSums(is.na(df))

# Drop ID-like columns
df$X  <- NULL
df$id <- NULL

# Diagnosis as factor (M = malignant, B = benign)
df$diagnosis <- factor(df$diagnosis, levels = c("M", "B"))

# Check duplicates
sum(duplicated(df))

# Outlier counts (z-score > 3)
numeric_vars <- sapply(df, is.numeric)
z <- scale(df[, numeric_vars])
outlier_counts <- colSums(abs(z) > 3)
outlier_counts

# (Optional) Histograms: saved to PDF
pdf("histograms.pdf", width = 12, height = 10)
par(mfrow = c(5, 6), mar = c(2, 2, 2, 1))
for (v in names(df)[numeric_vars]) {
  hist(df[[v]], main = v, col = "purple")
}
dev.off()
par(mfrow = c(1, 1))


#####################
# Train–Test Split
#####################

set.seed(123)

idx   <- sample(1:nrow(df), 0.7 * nrow(df))
train <- df[idx, ]
test  <- df[-idx, ]


############
# Scaling
############

numeric_vars <- sapply(train, is.numeric)

train_scaled <- train
test_scaled  <- test

# Save scaling parameters from training set
scaled_train_obj <- scale(train[, numeric_vars])
train_scaled[, numeric_vars] <- scaled_train_obj

test_scaled[, numeric_vars] <- scale(
  test[, numeric_vars],
  center = attr(scaled_train_obj, "scaled:center"),
  scale  = attr(scaled_train_obj, "scaled:scale")
)


################
# Linear SVM
################

tuned_linear <- tune.svm(
  diagnosis ~ .,
  data   = train_scaled,
  kernel = "linear",
  cost = c(0.1, 1, 2, 5, 10, 20, 30, 40)  
)

svm_linear <- tuned_linear$best.model

# Extract tuning results
linear_results <- tuned_linear$performances

# Convert error to accuracy
linear_results$Accuracy <- 1 - linear_results$error

# Plot of cross-validation with Accuracy
ggplot(linear_results, aes(x = cost, y = Accuracy)) +
  geom_line(color = "purple", size = 1.2) +
  geom_point(color = "purple", size = 3) +
  theme_minimal() +
  labs(
    title = "Linear SVM Tuning Results (C vs Accuracy)",
    x = "Cost (C)",
    y = "CV Accuracy"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Predictions
pred_linear_train <- predict(svm_linear, train_scaled)
pred_linear_test  <- predict(svm_linear, test_scaled)

pred_linear_test <- factor(pred_linear_test, levels = c("B", "M"))
test$diagnosis   <- factor(test$diagnosis,   levels = c("B", "M"))

## Confusion matrix
cm_linear <- confusionMatrix(pred_linear_test, test$diagnosis, positive = "M")
cm_linear$table

# heatmap
cm_linear_df <- as.data.frame(cm_linear$table)
colnames(cm_linear_df) <- c("Predicted", "Actual", "Freq")

ggplot(cm_linear_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "purple") +
  theme_minimal() +
  labs(title = "Confusion Matrix – Linear SVM (Test Set)")+
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))


# Metrics table
train_acc_linear <- mean(pred_linear_train == train$diagnosis)

linear_stats <- data.frame(
  `Train Accuracy`    = train_acc_linear,
  `Test Accuracy`     = cm_linear$overall["Accuracy"],
  Kappa               = cm_linear$overall["Kappa"],
  Sensitivity         = cm_linear$byClass["Sensitivity"],
  Specificity         = cm_linear$byClass["Specificity"],
  `Balanced Accuracy` = cm_linear$byClass["Balanced Accuracy"]
)

kable(linear_stats, digits = 3,
      caption = "Linear SVM Performance Metrics (70/30 Split)") %>%
  kable_styling(full_width = FALSE, position = "center")


##########################
# Radial SVM (Default)
##########################

svm_radial <- svm(
  diagnosis ~ .,
  data   = train_scaled,
  kernel = "radial",
  cost   = 1,
  gamma  = 1 / ncol(train_scaled)   # default-ish starting point
)

# Predictions
pred_radial_train <- predict(svm_radial, train_scaled)
pred_radial_test  <- predict(svm_radial, test_scaled)

pred_radial_test <- factor(pred_radial_test, levels = c("B", "M"))
test$diagnosis   <- factor(test$diagnosis,   levels = c("B", "M"))

# Confusion matrix
cm_radial <- confusionMatrix(pred_radial_test, test$diagnosis, positive = "M")

# Heatmap from confusion Matrix table
cm_radial_df <- as.data.frame(cm_radial$table)
colnames(cm_radial_df) <- c("Predicted", "Actual", "Freq")

ggplot(cm_radial_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "purple") +
  theme_minimal() +
  labs(title = "Confusion Matrix – Radial SVM (Default, Test Set)") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))

# Metrics table
train_acc_radial <- mean(pred_radial_train == train$diagnosis)

radial_stats <- data.frame(
  `Train Accuracy`    = train_acc_radial,
  `Test Accuracy`     = cm_radial$overall["Accuracy"],
  Kappa               = cm_radial$overall["Kappa"],
  Sensitivity         = cm_radial$byClass["Sensitivity"],
  Specificity         = cm_radial$byClass["Specificity"],
  `Balanced Accuracy` = cm_radial$byClass["Balanced Accuracy"]
)

kable(radial_stats, digits = 3,
      caption = "Radial SVM (Default) Performance Metrics (70/30 Split)") %>%
  kable_styling(full_width = FALSE, position = "center")



#######################################
# Hyperparameter Tuning (Radial SVM)
#######################################

set.seed(123)

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary  # ROC, Sens, Spec
)

# Grid of C and sigma (gamma)
grid <- expand.grid(
  C     = c(0.1, 1 , 2, 5, 10, 20, 30, 40),
  sigma = c(0.001, 0.01, 0.05, 0.1, 0.2)
)

svm_tuned <- train(
  diagnosis ~ .,
  data      = train_scaled,
  method    = "svmRadial",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = grid
)

svm_tuned
plot(svm_tuned)  # Cross-validation ROC vs (C, sigma)

#better table

svmgrid <- svm_tuned$results

ggplot(svmgrid, aes(x = C, y = ROC, color = factor(sigma))) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(
    name = "Sigma",
    values = c(
      "0.001" = "#DDD6FE",  
      "0.01"  = "#7C3AED",   
      "0.05"  = "#EC4899",   
      "0.1"   = "#FBCFE8",   
      "0.2"   = "#C026D3"    
      
    )
  ) +
  theme_minimal(base_size = 14) +
  labs(
    title = "SVM Radial — ROC by Cost and Sigma",
    x = "Cost (C)",
    y = "ROC (Repeated CV)"
  )


# Predictions
pred_tuned_train <- predict(svm_tuned, train_scaled)
pred_tuned_test  <- predict(svm_tuned, test_scaled)

# Force same level order
pred_tuned_test <- factor(pred_tuned_test, levels = c("B", "M"))
test$diagnosis   <- factor(test$diagnosis,   levels = c("B", "M"))

# Confusion matrix
cm_tuned <- confusionMatrix(pred_tuned_test, test$diagnosis, positive = "M")

# Heatmap from confusion Matrix table
cm_tuned_df <- as.data.frame(cm_tuned$table)
colnames(cm_tuned_df) <- c("Predicted", "Actual", "Freq")

ggplot(cm_tuned_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "purple") +
  theme_minimal() +
  labs(title = "Confusion Matrix – Radial SVM (Tuned, Test Set)") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))

# Metrics table (tuned model)
train_acc_tuned <- mean(pred_tuned_train == train$diagnosis)

tuned_stats <- data.frame(
  `Train Accuracy`    = train_acc_tuned,
  `Test Accuracy`     = cm_tuned$overall["Accuracy"],
  Kappa               = cm_tuned$overall["Kappa"],
  Sensitivity         = cm_tuned$byClass["Sensitivity"],
  Specificity         = cm_tuned$byClass["Specificity"],
  `Balanced Accuracy` = cm_tuned$byClass["Balanced Accuracy"]
)

kable(tuned_stats, digits = 3,
      caption = "Radial SVM (Tuned) Performance Metrics (70/30 Split)") %>%
  kable_styling(full_width = FALSE, position = "center")

#####################################
# 2D Visualization of Radial SVM
#####################################

# Use only two predictors for a 2D decision boundary
vars2    <- c("radius_mean", "concavity_mean")
train_2d <- train_scaled[, c(vars2, "diagnosis")]

svm_2d <- svm(
  diagnosis ~ .,
  data   = train_2d,
  kernel = "radial",
  cost   = 10,
  gamma  = 0.01
)

# Grid over feature space
x_range <- seq(min(train_2d$radius_mean), max(train_2d$radius_mean), length = 200)
y_range <- seq(min(train_2d$concavity_mean), max(train_2d$concavity_mean), length = 200)
grid <- expand.grid(radius_mean = x_range, concavity_mean = y_range)

# Predictions on grid
grid$pred <- predict(svm_2d, newdata = grid)

# Plot decision boundary
ggplot() +
  geom_tile(data = grid,
            aes(x = radius_mean, y = concavity_mean, fill = pred),
            alpha = 0.25) +
  geom_point(data = train_2d,
             aes(x = radius_mean, y = concavity_mean, color = diagnosis),
             size = 2) +
  scale_fill_manual(values = c("pink", "purple")) +
  scale_color_manual(values = c("pink4", "purple4")) +
  theme_minimal() +
  labs(title = "SVM Decision Boundary (2D Projection)",
       x = "Radius (mean)",
       y = "Concavity (mean)") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))


#############################################
# Variable Importance – Radial SVM (tuned)
#############################################

# Get importance object
radial_imp <- varImp(svm_tuned, scale = TRUE)

# Turn into data frame
radial_imp_df <- as.data.frame(radial_imp$importance)
radial_imp_df$Variable <- rownames(radial_imp_df)

score_col <- setdiff(names(radial_imp_df), "Variable")[1]

# Rename that column to "Importance"
radial_imp_df$Importance <- radial_imp_df[[score_col]]

# Sort descending by Importance
radial_imp_df <- radial_imp_df[order(-radial_imp_df$Importance), ]

# Print full ranking in console
print(radial_imp_df[, c("Variable", "Importance")])

# Plot top 20 variables
top_n <- 20

ggplot(head(radial_imp_df, top_n),
       aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "purple") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Variable Importance – Radial SVM (Tuned)",
    x = "Variable",
    y = "Importance"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))


#############################################
# Logistic regression for comparison
#############################################
set.seed(123)
log_model <- train(
  diagnosis ~ .,
  data = train_scaled,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)
log_model

pred_log <- predict(log_model, newdata = test_scaled)
cm_log <- confusionMatrix(pred_log, test_scaled$diagnosis)
cm_log

# Logistic Regression train accuracy
pred_log_train <- predict(log_model, newdata = train_scaled)
cm_log_train <- confusionMatrix(pred_log_train, train_scaled$diagnosis)
train_acc_log <- cm_log_train$overall["Accuracy"]

#############################################
# Final Comparison of the Models
#############################################
compare_stats <- data.frame(
  Model               = c("Linear SVM",
                          "Radial SVM (Default)",
                          "Radial SVM (Tuned)",
                          "Logistic Regression"),
  `Train Accuracy`    = c(train_acc_linear,
                          train_acc_radial,
                          train_acc_tuned,
                          train_acc_log),
  `Test Accuracy`     = c(cm_linear$overall["Accuracy"],
                          cm_radial$overall["Accuracy"],
                          cm_tuned$overall["Accuracy"],
                          cm_log$overall["Accuracy"]),
  Kappa               = c(cm_linear$overall["Kappa"],
                          cm_radial$overall["Kappa"],
                          cm_tuned$overall["Kappa"],
                          cm_log$overall["Kappa"]),
  Sensitivity         = c(cm_linear$byClass["Sensitivity"],
                          cm_radial$byClass["Sensitivity"],
                          cm_tuned$byClass["Sensitivity"],
                          cm_log$byClass["Sensitivity"]),
  Specificity         = c(cm_linear$byClass["Specificity"],
                          cm_radial$byClass["Specificity"],
                          cm_tuned$byClass["Specificity"],
                          cm_log$byClass["Specificity"]),
  `Balanced Accuracy` = c(cm_linear$byClass["Balanced Accuracy"],
                          cm_radial$byClass["Balanced Accuracy"],
                          cm_tuned$byClass["Balanced Accuracy"],
                          cm_log$byClass["Balanced Accuracy"])
)
kable(compare_stats,
      digits = 3,
      caption = "Model Performance Comparison (SVMs vs Logistic Regression)") %>%
  kable_styling(full_width = FALSE, position = "center")


#############################################
# Prediction of SVM Radial Tuned
#############################################

# 6 variables used (most interpreteble)
vars6 <- c(
  "radius_mean", "texture_mean","area_mean",
  "smoothness_mean", "compactness_mean", "concavity_mean")


# Keep diagnosis + these predictors
train_app <- train[, c("diagnosis", vars6)]

# 10-fold CV setup
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary  # ROC, Sens, Spec
)

# Grid for C and sigma (gamma)
grid <- expand.grid(
  C     = c(0.1,1, 2, 5, 10, 20, 30, 40),
  sigma = c(0.001, 0.01, 0.05, 0.1, 0.2)
)

set.seed(123)
svm_app <- train(
  diagnosis ~ .,
  data       = train_app,
  method     = "svmRadial",
  trControl  = ctrl,
  metric     = "ROC",
  tuneGrid   = grid,
  preProcess = c("center", "scale")   # caret stores scaling inside the model
)

# Inspect best parameters 
svm_app

# Save model for Shiny app
saveRDS(svm_app, "svm_shiny_model.rds")






