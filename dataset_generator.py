import pandas as pd
import numpy as np

def generator():
	# Set seed for reproducibility
	np.random.seed(42)

	# Generate 100 random ages between 18 and 60
	ages = np.random.randint(18, 60, 100)

	# Generate annual income between 20,000 and 120,000
	incomes = np.random.randint(20000, 120000, 100)

	# Generate purchase decision based on some probability (1 = Purchased, 0 = Not Purchased)
	purchase = np.array([1 if age > 30 and income > 50000 else 0 for age, income in zip(ages, incomes)])

	# Create DataFrame
	df = pd.DataFrame({'Age': ages, 'Annual_Income': incomes, 'Purchased': purchase})

	# Save as CSV
	file_path = "smartphone_purchase_data.csv"
	df.to_csv(file_path, index=False)

	# Return the file path
	return file_path

if __name__=="__main__":
	generator()