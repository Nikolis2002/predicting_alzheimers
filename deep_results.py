from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["alzheimers"]  
collection = db["deep_v2"]  

# Fetch documents
results = list(collection.find())

# Parse architecture string into list of integers
def parse_arch(arch_str):
    try:
        return [int(n) for n in arch_str.split(',')]
    except:
        return []

# Classify architecture shape
def get_strategy(neurons):
    if all(x > y for x, y in zip(neurons, neurons[1:])):
        return "pyramid"
    elif all(x < y for x, y in zip(neurons, neurons[1:])):
        return "inverted"
    elif all(x == y for x, y in zip(neurons, neurons[1:])):
        return "flat"
    elif len(neurons) == 3 and neurons[0] < neurons[1] > neurons[2] and neurons[0] != neurons[2]:
        return "sandwich"
    else:
        return "irregular"

# Build structured list for DataFrame
data = []
for doc in results:
    architecture = doc.get("choosen architecture", "")
    arch_list = parse_arch(architecture)
    if not arch_list:
        continue

    total_neurons = sum(arch_list)
    num_layers = len(arch_list)  # <-- number of hidden layers (2 or 3)
    strategy = get_strategy(arch_list)
    run_epochs = doc.get("params", {}).get("run_epochs", 0)
    accuracy = doc.get("Average Accuracy", None)
    mse = doc.get("Average MSE", None)
    loss = doc.get("Average loss", None)

    data.append({
        "Strategy": strategy,
        "Architecture": architecture,
        "Num Layers": num_layers,
        "Total Neurons": total_neurons,
        "Run Epochs": run_epochs,
        "Accuracy": accuracy,
        "MSE": mse,
        "Loss": loss
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# ---- 📊 PLOTS ----

# ✅ Plot: Accuracy by Strategy
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="Total Neurons", y="Accuracy", hue="Strategy", marker="o")
plt.title("Accuracy vs Total Neurons by Strategy")
plt.xlabel("Total Neurons")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Plot: Accuracy by Number of Layers (2 vs 3)
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Num Layers", y="Accuracy")
plt.title("Accuracy Distribution by Layer Count")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# ✅ Plot: Run Epochs vs Accuracy, colored by Num Layers
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Run Epochs", y="Accuracy", hue="Num Layers", style="Strategy", s=100)
plt.title("Run Epochs vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Optional: Print Top 5 Best-Performing Architectures
top = df.sort_values(by="Accuracy", ascending=False).head(5)
print("\n🏆 Top 5 Best Architectures:")
print(top[["Architecture", "Strategy", "Num Layers", "Run Epochs", "Accuracy"]])
