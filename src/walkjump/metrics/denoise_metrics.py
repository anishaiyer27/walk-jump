import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate SSIM
def evaluate_denoising(noisy_sample, denoised_sample):
    # Ensure the samples are in the proper range (e.g., [0, 1] for floating point)
    # You can also ensure they are in the expected range for your particular data
    data_range = noisy_sample.max() - noisy_sample.min()

    # Calculate SSIM between noisy and denoised samples
    ssim_value = ssim(noisy_sample, denoised_sample, data_range=data_range)
    return ssim_value

# Function to evaluate the performance of the denoising model
def evaluate_denoising_model(noisy_samples, denoised_samples):
    # Ensure both inputs are numpy arrays and have the same shape
    assert noisy_samples.shape == denoised_samples.shape, "Samples must have the same shape."
    
    # Initialize lists to store results
    ssim_scores = []
    
    # Iterate over each sample and compute SSIM
    for noisy, denoised in zip(noisy_samples, denoised_samples):
        ssim_value = evaluate_denoising(noisy, denoised)
        ssim_scores.append(ssim_value)
    
    # Convert list to numpy array
    ssim_scores = np.array(ssim_scores)
    
    # Calculate the mean and standard deviation of SSIM
    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Standard Deviation of SSIM: {std_ssim:.4f}")

    # Plot the distribution of SSIM scores
    plt.figure(figsize=(8, 6))
    plt.hist(ssim_scores, bins=20, color='blue', alpha=0.7)
    plt.title("Distribution of SSIM Scores for Denoised Samples")
    plt.xlabel("SSIM Value")
    plt.ylabel("Frequency")
    plt.show()

# Loading the noisy and denoised samples
def load_samples():
    # Load noisy samples (replace with actual file path)
    noisy_samples_path = "./data/poas.csv.gz"
    noisy_samples_df = pd.read_csv(noisy_samples_path)

    # Load denoised samples (replace with actual file path)
    denoised_samples_path = "./data/samples_2denoise.csv"
    denoised_samples_df = pd.read_csv(denoised_samples_path)

    # Convert the DataFrames to numpy arrays
    denoised_samples = denoised_samples_df.to_numpy()
    noisy_samples = noisy_samples_df.to_numpy()
    noisy_samples = noisy_samples[np.random.choice(noisy_samples.shape[0], len(denoised_samples), replace=False)]
    
    print(noisy_samples, denoised_samples)

    return noisy_samples, denoised_samples

# Main function to run the evaluation
def main():
    # Load the noisy and denoised samples
    noisy_samples, denoised_samples = load_samples()

    # Evaluate the denoising performance
    evaluate_denoising_model(noisy_samples, denoised_samples)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
