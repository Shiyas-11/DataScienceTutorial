import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Define the transaction dataset (One-Hot Encoded Format)
dataset = [
    ['milk', 'bread', 'butter','egg'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'egg','butter'],
    ['bread', 'butter', 'cheese'],
    ['milk', 'bread','egg', 'cheese'],
    ['milk', 'egg','cheese']
]

# Convert dataset into a DataFrame
items = sorted(set(item for transaction in dataset for item in transaction))  # Unique items
df = pd.DataFrame([{item: (item in transaction) for item in items} for transaction in dataset])

# Step 2: Apply Apriori Algorithm
min_support = 0.3  # Minimum support threshold (30%)
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Step 3: Generate Association Rules
min_confidence = 0.6  # Minimum confidence threshold (60%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display Results
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
