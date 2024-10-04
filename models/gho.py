# Grass Hopper Algorithm

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

# Define the number of grasshoppers and maximum number of iterations
num_grasshoppers = 10
max_iter = 100

# Define the initial population of grasshoppers
pop = {}
for i in range(num_grasshoppers):
    pop[i] = {param: np.random.choice(search_space[param]) for param in search_space}

# Define the fitness function
def fitness(params):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

# Define the distance function
def distance(x, y):
    return np.sqrt(np.sum((np.array(list(x.values())) - np.array(list(y.values()))) ** 2))

# Run the optimization loop
for t in range(max_iter):
    # Evaluate the fitness of each grasshopper
    fitness_scores = {i: fitness(pop[i]) for i in range(num_grasshoppers)}

    # Sort the grasshoppers based on their fitness scores
    sorted_pop = sorted(pop.items(), key=lambda x: -fitness_scores[x[0]])

    # Calculate the mean and standard deviation of the fitness scores
    fitness_mean = np.mean(list(fitness_scores.values()))
    fitness_std = np.std(list(fitness_scores.values()))

    # Update the position of each grasshopper
    for i in range(num_grasshoppers):
        # Calculate the distance to each other grasshopper
        distances = [distance(pop[i], pop[j]) for j in range(num_grasshoppers) if i != j]
        print(len(distances))
        print(num_grasshoppers)
        # Calculate the influence of each other grasshopper
        influences = [fitness_scores[j] / (distances[j] + 1e-6) for j in range(num_grasshoppers) if i != j]

        # Calculate the direction of movement
        direction = np.sum([influences[j] * (np.array(list(pop[j].values())) - np.array(list(pop[i].values()))) for j in range(num_grasshoppers) if i != j], axis=0)

        # Update the position of the grasshopper
        pop[i] = {param: max(min(int(pop[i][param] + np.random.normal() * direction[k]), search_space[param][-1]), search_space[param][0]) for k, param in enumerate(search_space)}

# Find the best hyperparameters
best_params = sorted_pop[0][1]

# Train the final model with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)