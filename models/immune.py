import numpy as np
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define the affinity function
def affinity(X_train, y_train, params):
    # Create LightGBM classifier with specified hyperparameters
    # print(params)
    params = {key: value for key, value in params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}

    clf = lgb.LGBMClassifier(**params)

    # Train and evaluate model on training set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Compute affinity as inverse of classification error
    return 1 / (1 + accuracy)


# Define the immune optimization algorithm
def immune_optimization(X_train, y_train, param_ranges, num_ants=10, num_generations=100, mutation_rate=0.1):
    # Initialize population of antibodies
    num_params = len(param_ranges)
    population = [{} for i in range(num_ants)]
    for i in range(num_ants):
        for p in param_ranges.keys():
            population[i][p] = np.random.choice(param_ranges[p])

    # Iterate over generations
    best_params = None
    best_affinity = 0
    for gen in range(num_generations):
        # Compute affinity of each antibody
        affinities = [affinity(X_train, y_train, population[i]) for i in range(num_ants)]
        if max(affinities) > best_affinity:
            best_params = population[np.argmax(affinities)]
            best_affinity = max(affinities)

        # Select antibodies for cloning based on affinity
        clone_probs = affinities / np.sum(affinities)
        num_clones = int(np.floor(num_ants / 2))
        clones = []
        for i in range(num_clones):
            parent_idx = np.random.choice(range(num_ants), p=clone_probs)
            clones.append(population[parent_idx])

        # Mutate clones by randomly selecting a parameter and perturbing its value
        for clone in clones:
            for p, p_ in zip(param_ranges.keys(), range(num_params)):
                if np.random.uniform() < mutation_rate:
                    clone[p_] = np.random.choice(param_ranges[p])

        # Replace least-affine antibodies with clones
        replace_indices = np.argsort(affinities)[:num_clones]
        for i in range(num_clones):
            population[replace_indices[i]] = clones[i]

    return best_params, best_affinity


# Define the hyperparameter ranges to be tuned
param_ranges = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [10, 20, 30]
}

# Run the Immune Optimization Algorithm
best_params, best_affinity = immune_optimization(X_train, y_train, param_ranges)
best_params = {key: value for key, value in best_params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}

# Create and evaluate the LightGBM classifier with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)
