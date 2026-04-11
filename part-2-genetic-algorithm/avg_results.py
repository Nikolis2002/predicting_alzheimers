import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    col = client['genetic_algos']['resultsv2']

    summary = []

    # Process each configuration row
    for row in range(1, 11):
        # Build tags for this row
        tags = [f"row{row}_run{run}" for run in range(1, 11)]
        docs = list(col.find({"params.tag": {"$in": tags}}))

        if not docs:
            print(f"No documents found for row {row}. Skipping.")
            continue

        # Extract metrics
        best_fits = [doc['best_fitness'] for doc in docs]
        gens      = [doc['generations']   for doc in docs]
        histories = [doc['history']       for doc in docs]

        # Compute averages
        avg_best = np.mean(best_fits)
        avg_gens = np.mean(gens)
        summary.append({
            "Config Row": row,
            "Avg Best Fitness": avg_best,
            "Avg Generations": avg_gens
        })

        # Align histories by padding with the last value
        max_len = max(len(h) for h in histories)
        aligned = [
            h + [h[-1]] * (max_len - len(h)) if len(h) < max_len else h
            for h in histories
        ]
        avg_history = np.mean(aligned, axis=0)

        # Plot evolution curve
        plt.figure()
        plt.plot(range(1, len(avg_history) + 1), avg_history)
        plt.title(f"Avg Evolution Curve — Row {row}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.tight_layout()
        plt.savefig(f"avg_curve_row{row}.png")
        plt.close()
        print(f"Saved plot avg_curve_row{row}.png")

    # Create summary DataFrame
    df = pd.DataFrame(summary)
    # Print to console
    print("\\nOverall Summary:")
    print(df.to_string(index=False))

    df.to_csv("summary_results.csv", index=False)
    print("Summary saved to 'summary_results.csv'")

if __name__ == "__main__":
    main()