import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Generate "real" data (e.g., ages of patients in a study)
np.random.seed(42)
real_data = np.random.normal(loc=45, scale=15, size=1000)

# 2. Generate "synthetic" data with the same statistical properties
synthetic_data = np.random.normal(loc=np.mean(real_data), scale=np.std(real_data), size=1000)

# 3. Evaluate the distributions using a statistical test
stat, p_value = stats.ks_2samp(real_data, synthetic_data)
print(f"Kolmogorov-Smirnov Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Result: The synthetic data follows the same distribution as the real data.")
else:
    print("Result: The distributions are significantly different.")

# 4. Visualize the distributions
plt.figure(figsize=(8, 5))
plt.hist(real_data, bins=30, alpha=0.5, label='Real Data', color='blue', density=True)
plt.hist(synthetic_data, bins=30, alpha=0.5, label='Synthetic Data', color='orange', density=True)
plt.title('Comparison of Real vs. Synthetic Data Distributions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('distribution_comparison.png')
print("Chart saved as distribution_comparison.png")
