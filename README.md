# Alzheimer’s Disease Prediction & Feature-Selection GA

## Part 1: Fine-Tuning a Neural Network for Alzheimer’s Prediction

This section implements the fine-tuning of a neural network to predict Alzheimer’s disease. The repository structure is:

- **`alzheimers_prediction.py`**  
  The main script: performs hyperparameter search and fine-tuning of a single-hidden-layer network.

- **`run.py`**  
  Launches experiments with fixed hyperparameters on the one-layer model.

- **`results.py`**  
  Connects to MongoDB, retrieves stored metrics, and computes summary statistics to compare against the plots.

- **`deep_run.py`**  
  Runs the same experiments as `run.py`, but for architectures with multiple hidden layers.

- **`deep_results.py`**  
  Processes and visualizes the MongoDB results for the multi-layer experiments.

- **`screenshots/`**  
  Contains loss & accuracy plots for single-layer and multi-layer models using **L2** regularization.

- **`screenshots_l1/`**  
  Contains the corresponding plots when using **L1** regularization.

> **Note:** A running MongoDB instance with the experiment data is required to reproduce the metric-vs-plot comparisons.

---

## Part 2: Genetic Algorithm for Feature Selection

All Part 2 code is in the `Part2/` folder:

- **`genetic_algo.py`**  
  Core implementation of the genetic algorithm for feature selection. (Full, commented source is in the repo.)

- **`run_experiments.py`**  
  Automates running `genetic_algo.py` over all desired hyperparameter combinations.

- **`avg_results.py`**  
  Fetches experiment results from MongoDB and generates plots for each table row.

- **`compare_nns.py`**  
  Retrains the neural network on the selected feature subsets. (See the “Evaluation” section of the report for details.)



> **Note:** For examiner:If similar code exists you can check the commit history to see that this code is the orginal,also last 3 hours the repository became private for same reasons.
