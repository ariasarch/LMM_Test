### Quick LMM Model ###
### By Ari Asarch   ###
### 11.16.2023      ###

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

# Load your data
# data = pd.read_csv('your_data.csv')  # Assuming your data is in a CSV file

# Setting random seed for reproducibility
np.random.seed(0)

# Creating a more diverse and larger dataset
n_subjects = 40  # Number of subjects
n_timepoints = 2  # Number of time points

data = pd.DataFrame({
    'SubjectID': np.repeat(np.arange(1, n_subjects + 1), n_timepoints),
    'Sex': np.random.choice(['Male', 'Female'], n_subjects * n_timepoints),
    'Blast': np.random.choice(['Yes', 'No'], n_subjects * n_timepoints),
    'DrugTreatment': np.random.choice(['Treated', 'Control'], n_subjects * n_timepoints),
    'FentanylIntake': np.random.uniform(5.0, 15.0, n_subjects * n_timepoints),
    'TimePoint': np.tile([1, 2], n_subjects)
})

data.head(10)  # Displaying the first 10 rows for a preview

# Model Specification
md = smf.mixedlm("FentanylIntake ~ Sex * Blast * DrugTreatment", data, 
                 groups=data["SubjectID"], re_formula="~TimePoint")
mdf = md.fit()

# Print the summary
print(mdf.summary())

# Extracting coefficients, standard errors, and p-values from the model summary
coefs = mdf.params
stderrs = mdf.bse
pvalues = mdf.pvalues

# Create arrays from the dictionaries
df_results = pd.DataFrame({'Coefficients': coefs, 'StdErr': stderrs, 'pvalues': pvalues})
df_results = df_results.drop(['Group Var', 'Group x TimePoint Cov', 'TimePoint Var'])

# Plotting the coefficients
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(df_results.index, df_results['Coefficients'], 
              yerr=1.96*df_results['StdErr'], 
              color='blue', 
              alpha=0.7)

# Add a line at 0 for reference
ax.axhline(0, color='black', linestyle='--')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.xlabel('Variables')
plt.ylabel('Coefficient Value')
plt.title('Coefficients and Significance in the Mixed Linear Model')

# Adding a star for significant coefficients
for i, pval in enumerate(df_results['pvalues']):
    if pval < 0.05:
        # Place a star above the corresponding bar
        ax.text(i, df_results['Coefficients'][i] + 2*df_results['StdErr'][i], '*', 
                ha='center', va='bottom', color='red', fontsize=14)

plt.tight_layout()
plt.show()

# Plotting residuals
# residuals = mdf.resid
# plt.hist(residuals, bins=20)
# plt.title('Residual Distribution')
# plt.xlabel('Residuals')
# plt.ylabel('Frequency')
# plt.show()