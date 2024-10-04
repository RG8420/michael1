# Coral Reef Optimization

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the search space for hyperparameters
search_space = {
    "num_leaves": np.arange(10, 100),
    "max_depth": np.arange(3, 20),
    "learning_rate": np.logspace(-3, 0, 100)
}

# Define the number of coral reefs and maximum number of iterations
num_reefs = 10
max_iter = 100

# Define the initial population of coral reefs
pop = {}
for i in range(num_reefs):
    pop[i] = {param: np.random.choice(search_space[param]) for param in search_space}

# Define the fitness function
def fitness(params):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

# Run the optimization loop
for t in range(max_iter):
    # Evaluate the fitness of each coral reef
    fitness_scores = {i: fitness(pop[i]) for i in range(num_reefs)}

    # Sort the coral reefs based on their fitness scores
    sorted_pop = sorted(pop.items(), key=lambda x: -fitness_scores[x[0]])

    # Calculate the mean and standard deviation of the fitness scores
    fitness_mean = np.mean(list(fitness_scores.values()))
    fitness_std = np.std(list(fitness_scores.values()))

    # Calculate the probability of a coral reef being selected for reproduction
    probs = [1.0 / (1.0 + np.exp(-2.0 * (fitness_scores[i] - fitness_mean) / fitness_std)) for i in range(num_reefs)]
    probs = [p / sum(probs) for p in probs]

    # Create a new population of coral reefs through reproduction and mutation
    new_pop = {}
    for i in range(num_reefs):
        # Select two parent coral reefs
        parent1_idx = np.random.choice(range(num_reefs), p=probs)
        parent2_idx = np.random.choice(range(num_reefs), p=probs)

        # Perform crossover
        new_params = {}
        for param in search_space:
            if np.random.rand() < 0.5:
                new_params[param] = pop[parent1_idx][param]
            else:
                new_params[param] = pop[parent2_idx][param]

        # Perform mutation
        for param in search_space:
            if np.random.rand() < 0.1:
                new_params[param] = np.random.choice(search_space[param])

        # Add the new coral reef to the population
        new_pop[i] = new_params

    # Replace the old population with the new population
    pop = new_pop

# Find the best hyperparameters
best_params = sorted_pop[0][1]

# Train the final model with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)
