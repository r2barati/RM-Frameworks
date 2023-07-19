```R
# Load required packages
install.packages("rpart")
library(rpart)

# Sample dataset for Commercial Real Estate Loans
cre_loans_data <- data.frame(
  loan_amount = c(100000, 200000, 300000, 150000, 250000),
  loan_to_value = c(0.7, 0.8, 0.6, 0.75, 0.9),
  debt_service_coverage = c(1.2, 1.1, 1.3, 1.0, 1.4),
  credit_score = c(700, 750, 680, 650, 720),
  good_loan = c(1, 1, 0, 0, 1) # Target variable: 1 for good loan, 0 for bad loan
)

# Prepare the formula for the decision tree
formula <- good_loan ~ loan_amount + loan_to_value + debt_service_coverage + credit_score

# Build the decision tree model
cre_tree <- rpart(formula, data = cre_loans_data, method = "class")

# Plot the decision tree
plot(cre_tree)
text(cre_tree, use.n = TRUE, all = TRUE, cex = 0.8)
```

In this example, we have created a simple sample dataset cre_loans_data with five records, each containing features such as loan_amount, loan_to_value, debt_service_coverage, credit_score, and the target variable good_loan indicating whether the loan performance is good (1) or bad (0).

Please note that this is a simplified and synthetic dataset for demonstration purposes. In a real-world scenario, you would use actual data from a commercial real estate loan portfolio, which might involve hundreds or thousands of records with more diverse and informative features. The quality and size of the dataset will significantly impact the decision tree's effectiveness and generalizability. Additionally, you should always validate and prune the decision tree to optimize its performance.
