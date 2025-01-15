import torch
from sklearn.neighbors import KernelDensity
from walkjump.conformity import conformity_score
import matplotlib.pyplot as plt
import pandas as pd

# Pick some reference distribution
mu = torch.zeros(5)
covariance_matrix = torch.eye(5)
reference_distribution = pd.read_csv("data/poas.csv.gz") # "read from file"
#reference_distribution = 
#reference_distribution = torch.distributions.MultivariateNormal(mu, covariance_matrix)

X_train = reference_distribution.sample(1000)
X_val = reference_distribution.sample(200)

# Evaluate on samples from uniform distribution
X_test_1 = pd.read_csv("data/samplesDD/description_heavy.csv")     # "dWJS on DD"
X_test_2 = pd.read_csv("data/samplesDD/description_light.csv") 
#X_test_2 = pd.read_csv("data/samplesED.csv")     # "dWJS on ED"
#X_test_3 = pd.read_csv("data/samplesEE.csv")     # "dWJS on EE"

# Fit a density estimator to the training data
kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(X_train)


# Get log probability of validation data and test data
log_prob_val = torch.from_numpy(kde.score_samples(X_val))

log_prob_test_1 = torch.from_numpy(kde.score_samples(X_test_1))
log_prob_test_2 = torch.from_numpy(kde.score_samples(X_test_2))
log_prob_test_3 = torch.from_numpy(kde.score_samples(X_test_3))


# Use validation log probabilities to compute conformity scores
conformity_test_1 = conformity_score(
    log_prob_test_1,
    log_prob_val
)

conformity_test_2 = conformity_score(
    log_prob_test_2,
    log_prob_val
)


# Plot distribution of conformity scores for each evaluated set
plt.violinplot([conformity_test_1, conformity_test_2])
#plt.violinplot([conformity_test_1, conformity_test_2, conformity_test_3])
plt.ylabel("Conformity (p-value)")
#plt.xticks([1, 2], ["#1 val vs DD", "#2 val vs ED", "#3 val vs EE"])
plt.xticks([1, 2], ["#1 val vs DD"])
plt.show()

# Mean conformity as a single statistic

#    - > 0.5: higher conformity, more similar to training data than validation data
#    - 0.5: optimal conformity, as on average, the test and validation data are equally likely under the reference distribution
#    - < 0.5: lower conformity, validation is more similar to training data than test data


mean_conformity_test_1 = conformity_test_1.mean()
mean_conformity_test_2 = conformity_test_2.mean()

print(f"Mean conformity for #1: {mean_conformity_test_1:.2f}")
print(f"Mean conformity for #2: {mean_conformity_test_2:.2f}")
