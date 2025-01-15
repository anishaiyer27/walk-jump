import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from walkjump.conformity import conformity_score
import matplotlib.pyplot as plt
import os

# Load the original data file
print(os.getcwd())  # Print current working directory to verify the file path
poas_path = "./data/poas.csv.gz"
poas_df = pd.read_csv(poas_path)

# Convert non-numeric values to NaN, then drop rows with NaN values
poas_df = poas_df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
poas_df = poas_df.dropna()  # Drop rows with NaN values (or handle them as needed)

# Check if X_train is empty after dropping NaN values
X_train = torch.tensor(poas_df.to_numpy(), dtype=torch.float32)
if X_train.size(0) == 0:
    print("X_train is empty after dropping NaN values.")

# Print the shape of X_train to verify
print(f"X_train shape: {X_train.shape}")

# If X_train is not empty, proceed with the rest of the code
if X_train.size(0) > 0:
    # Split the data into training and validation sets
    train_size = int(0.8 * len(X_train))
    X_train, X_val = X_train[:train_size], X_train[train_size:]

    # Load the resultant samples from samples.csv
    samples_path = "./data/samples_2denoise.csv"
    sample_df = pd.read_csv(samples_path)

    # Convert non-numeric values to NaN, then drop rows with NaN values
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
    sample_df = sample_df.dropna()  # Drop rows with NaN values (or handle them as needed)

    # Convert the sample data to a tensor
    X_samples = torch.tensor(sample_df.to_numpy(), dtype=torch.float32)

    # Generate additional test datasets
    # Example: Uniform distribution in the range of observed values in the data
    data_min = X_train.min(dim=0).values
    data_max = X_train.max(dim=0).values
    X_test_1 = X_train[torch.randint(0, len(X_train), (100,))]  # Random subset of training data
    X_test_2 = torch.rand(100, X_train.shape[1]) * (data_max - data_min) + data_min  # Uniform distribution

    # Fit a density estimator to the training data
    kde = KernelDensity(kernel="gaussian", bandwidth=0.2)
    kde.fit(X_train)

    # Get log probability of validation data, test data, and samples
    log_prob_val = torch.from_numpy(kde.score_samples(X_val))
    log_prob_test_1 = torch.from_numpy(kde.score_samples(X_test_1))
    log_prob_test_2 = torch.from_numpy(kde.score_samples(X_test_2))
    log_prob_samples = torch.from_numpy(kde.score_samples(X_samples))

    # Use validation log probabilities to compute conformity scores
    conformity_test_1 = conformity_score(log_prob_test_1, log_prob_val)
    conformity_test_2 = conformity_score(log_prob_test_2, log_prob_val)
    conformity_samples = conformity_score(log_prob_samples, log_prob_val)

    # Plot distribution of conformity scores for each evaluated set
    plt.violinplot([conformity_test_1, conformity_test_2, conformity_samples])
    plt.ylabel("Conformity (p-value)")
    plt.xticks(
        [1, 2, 3],
        ["Random Training Subset", "Uniform Distribution", "Samples"],
    )
    plt.title("Distribution of Conformity Scores")
    plt.show()  # Ensure the plot is displayed

    # Mean conformity as a single statistic
    #    - > 0.5: higher conformity, more similar to training data than validation data
    #    - 0.5: optimal conformity, as on average, the test and validation data are equally likely under the reference distribution
    #    - < 0.5: lower conformity, validation is more similar to training data than test data
    mean_conformity_test_1 = conformity_test_1.mean()
    mean_conformity_test_2 = conformity_test_2.mean()
    mean_conformity_samples = conformity_samples.mean()

    print(f"Mean conformity for Random Training Subset: {mean_conformity_test_1:.2f}")
    print(f"Mean conformity for Uniform Distribution: {mean_conformity_test_2:.2f}")
    print(f"Mean conformity for Samples: {mean_conformity_samples:.2f}")
else:
    print("X_train is empty; cannot proceed.")
