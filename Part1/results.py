import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math

# === 1. Connect to MongoDB and load data ===
client = MongoClient("mongodb://localhost:27017/")
db = client["alzheimers"]
collection = db["average_results"]

# Load data from MongoDB into a list of dicts
data = list(collection.find())

# Convert to DataFrame and flatten nested fields
df = pd.json_normalize(data)

# Rename fields for consistency
df.rename(columns={
    'params.optimizer': 'optimizer',
    'params.momentum': 'momentum',
    'params.learning rate': 'learning_rate',
    'params.epochs': 'epochs',
    'params.run_epochs': 'run_epochs',
    'params.number of hidden layers': 'num_hidden_layers',
    'params.hidden layer activation function': 'activation_func',
    'params.loss function': 'loss_func',
    'params.regulazation rate': 'reg_rate',
    'Average Accuracy': 'accuracy',
    'Average loss': 'loss',
    'Average MSE': 'mse',
    'use L2': 'use_l2',
    'use L1': 'use_l1',
}, inplace=True)

layer_mapping = {
    "same": 38,
    "double": 2*38,
    "two thirds": math.ceil(2*38/3),
    "half": math.ceil(38/2)
}

# Add this line BEFORE calling drop_duplicates
df['layer_count'] = df['num_hidden_layers'].apply(
    lambda x: layer_mapping.get(str(x).lower().strip(), 1)
)
# === 3. Plot 1: Accuracy vs Number of Hidden Layers (First entry per architecture) ===
first_per_arch = df.drop_duplicates(subset=['num_hidden_layers'])
layer_summary = first_per_arch.groupby('layer_count')['accuracy'].mean().reset_index()

plt.figure(figsize=(6, 4))
bars = plt.bar(layer_summary['layer_count'].astype(str), layer_summary['accuracy'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom')



plt.title('Accuracy by Number of Hidden Layers (First Entry per Architecture Type)')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Plot 2: Accuracy of all "double" entries by LR & Momentum ===
double_df = df[df['num_hidden_layers'] == 'double'].copy()

double_df['label'] = double_df.apply(
    lambda row: f"lr={row['learning_rate']}, m={row['momentum']}", axis=1
)
double_df.drop_duplicates(subset=['label'], inplace=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(double_df['label'], double_df['accuracy'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom')

plt.title('Accuracy for "double" Architectures by LR & Momentum')
plt.xlabel('Learning Rate / Momentum')
plt.ylabel('Average Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. Plot 3: Compare L1 and L2 regularization ===
l1_df = df[df['use_l1'] == True]
l2_df = df[df['use_l2'] == True]

labels = ['L1', 'L2']
accuracies = [
    l1_df['accuracy'].mean() if not l1_df.empty else 0,
    l2_df['accuracy'].mean() if not l2_df.empty else 0
]

plt.figure(figsize=(6, 4))
plt.bar(labels, accuracies, color=['orange', 'blue'])
plt.title('L1 vs L2 Regularization (Avg Accuracy)')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Print the best-performing experiment ===
# === Best L1 and L2 Results ===

# Best L1 result
if not df[df['use_l1'] == True].empty:
    best_l1 = df[df['use_l1'] == True].loc[df[df['use_l1'] == True]['accuracy'].idxmax()]

    print("\nBest L1  Result:")
    print(f"Accuracy: {best_l1['accuracy']:.4f}")
    print(f"Architecture: {best_l1.get('chosen_weight', 'N/A')}")
    print(f"Reg rate: {best_l1.get('reg_rate', 'N/A')}")
    print(f"Hidden Layer Strategy: {best_l1.get('params.number of hidden layers', 'N/A')}")
    print(f"Optimizer: {best_l1.get('optimizer', 'N/A')}")
    print(f"Learning Rate: {best_l1.get('learning_rate', 'N/A')}")
    print(f"Momentum: {best_l1.get('momentum', 'N/A')}")
    print(f" Regularization Rate: {best_l1.get('params.regulazation rate', 'N/A')}")
    print(f"✔ Average Loss: {best_l1.get('loss', 'N/A')}")
    print(f"✔ Average MSE: {best_l1.get('mse', 'N/A')}")

# Best L2 result
if not df[df['use_l2'] == True].empty:
    best_l2 = df[df['use_l2'] == True].loc[df[df['use_l2'] == True]['accuracy'].idxmax()]

    print("\nBest L2  Result:")
    print(f"Accuracy: {best_l2['accuracy']:.4f}")
    print(f"Architecture: {best_l2.get('chosen_weight', 'N/A')}")
    print(f"Reg rate: {best_l2.get('reg_rate', 'N/A')}")
    print(f"Hidden Layer Strategy: {best_l2.get('params.number of hidden layers', 'N/A')}")
    print(f"Optimizer: {best_l2.get('optimizer', 'N/A')}")
    print(f"Learning Rate: {best_l2.get('learning_rate', 'N/A')}")
    print(f"Momentum: {best_l2.get('momentum', 'N/A')}")
    print(f"Regularization Rate: {best_l2.get('params.regulazation rate', 'N/A')}")
    print(f"Average Loss: {best_l2.get('loss', 'N/A')}")
    print(f"Average MSE: {best_l2.get('mse', 'N/A')}")

