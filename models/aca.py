# Ant Colony Algorithm

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


# Define the pheromone update function
def update_pheromone(trails, ants, decay=0.1):
    trails *= (1 - decay)
    for ant in ants:
        for i in range(len(ant)):
            trails[i][int(ant[i])] += 1.0 / len(ant)


# Define the ant behavior function
def ant_behavior(X_train, y_train, params):
    # Create LightGBM classifier with specified hyperparameters
    clf = lgb.LGBMClassifier(**params)

    # Train and evaluate model on training set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    return accuracy


# Define the ant colony optimization function
def ant_colony_optimization(X_train, y_train, param_ranges, num_ants=10, num_generations=100, decay=0.1):
    # Initialize pheromone trails
    print(param_ranges)
    num_params = len(param_ranges)
    num_values = [len(param_ranges[p]) for p in param_ranges.keys()]
    trails = np.ones(num_values)

    # Iterate over generations
    best_params = None
    best_fitness = 0
    for gen in range(num_generations):
        # Generate candidate solutions (ants)
        ants = []
        for i in range(num_ants):
            # Construct a solution by choosing one value per parameter
            solution = [np.random.randint(num_values[p]) for p in range(len(num_values))]
            ants.append(solution)

        # Evaluate candidate solutions and update pheromone trails
        fitness_values = [ant_behavior(X_train, y_train, {p: param_ranges[p][int(ants[i][p_])] for p, p_ in zip(param_ranges.keys(), range(num_params))})
                          for i in range(num_ants)]
        if max(fitness_values) > best_fitness:
            best_params = {p: param_ranges[p][int(ants[np.argmax(fitness_values)][p_])] for p, p_ in zip(param_ranges.keys(), range(num_params))}
            best_fitness = max(fitness_values)
        update_pheromone(trails, ants, decay=decay)

    return best_params, best_fitness


# Define the hyperparameter ranges to be tuned
param_ranges = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [10, 20, 30]
}

# Run the Ant Colony Optimization algorithm
best_params, best_fitness = ant_colony_optimization(X_train, y_train, param_ranges)

# Create and evaluate the LightGBM classifier with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy: ", accuracy)
