"""
Script to download/generate sample datasets for testing external CSV functionality.
"""
import pandas as pd
import os

# Create test_data directory
os.makedirs("test_data", exist_ok=True)

# ============================================
# 1. Iris Dataset (Classification)
# ============================================
from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = [iris.target_names[i] for i in iris.target]

df_iris.to_csv("test_data/iris.csv", index=False)
print("✅ Created: test_data/iris.csv")
print(f"   Shape: {df_iris.shape}")
print(f"   Target: 'species' (classification)")
print(f"   Classes: {list(iris.target_names)}")

# ============================================
# 2. California Housing (Regression)
# ============================================
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['price'] = housing.target

# Take a sample for faster testing
df_housing_sample = df_housing.sample(n=1000, random_state=42)
df_housing_sample.to_csv("test_data/california_housing.csv", index=False)
print("\n✅ Created: test_data/california_housing.csv")
print(f"   Shape: {df_housing_sample.shape}")
print(f"   Target: 'price' (regression)")

# ============================================
# 3. Digits Dataset (Classification)
# ============================================
from sklearn.datasets import load_digits

digits = load_digits()
df_digits = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
df_digits['digit'] = digits.target

# Take a sample
df_digits_sample = df_digits.sample(n=500, random_state=42)
df_digits_sample.to_csv("test_data/digits.csv", index=False)
print("\n✅ Created: test_data/digits.csv")
print(f"   Shape: {df_digits_sample.shape}")
print(f"   Target: 'digit' (classification)")

print("\n" + "=" * 50)
print("📁 Test datasets created in 'test_data/' folder")
print("=" * 50)
