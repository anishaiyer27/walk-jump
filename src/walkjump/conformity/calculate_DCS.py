import torch

# Pick some reference distribution
mu = torch.zeros(5)
covariance_matrix = torch.eye(5)
reference_distribution = torch.distributions.MultivariateNormal(mu, covariance_matrix)

X_train = reference_distribution.sample((1000,))
X_val = reference_distribution.sample((200,))

# Evaluate on samples from uniform distribution
X_test_1 = reference_distribution.sample((100,))
X_test_2 = torch.rand(100, 5) * 5 - 2.5
# Fit a density estimator to the training data

from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(X_train)
# Get log probability of validation data and test data
log_prob_val = torch.from_numpy(kde.score_samples(X_val))

log_prob_test_1 = torch.from_numpy(kde.score_samples(X_test_1))
log_prob_test_2 = torch.from_numpy(kde.score_samples(X_test_2))
# Use validation log probabilities to compute conformity scores
from walkjump.conformity import conformity_score

conformity_test_1 = conformity_score(
    log_prob_test_1,
    log_prob_val
)

conformity_test_2 = conformity_score(
    log_prob_test_2,
    log_prob_val
)
# Plot distribution of conformity scores for each evaluated set

import matplotlib.pyplot as plt

plt.violinplot([conformity_test_1, conformity_test_2])
plt.ylabel("Conformity (p-value)")
plt.xticks([1, 2], ["#1 normal distribution (0, I)", "#2 uniform distribution [-2.5, 2.5]", ])
plt.show()

# Mean conformity as a single statistic

#    - > 0.5: higher conformity, more similar to training data than validation data
#    - 0.5: optimal conformity, as on average, the test and validation data are equally likely under the reference distribution
#    - < 0.5: lower conformity, validation is more similar to training data than test data


mean_conformity_test_1 = conformity_test_1.mean()
mean_conformity_test_2 = conformity_test_2.mean()

print(f"Mean conformity for #1: {mean_conformity_test_1:.2f}")
print(f"Mean conformity for #2: {mean_conformity_test_2:.2f}")

