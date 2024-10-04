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


# Define the fitness function
def fitness(X_train, y_train, params):
    # Create LightGBM classifier with specified hyperparameters
    params = {key: value for key, value in params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}
    params['max_depth'] = int(params['max_depth'])
    params['num_leaves'] = int(params['num_leaves'])
    # print(params)
    clf = lgb.LGBMClassifier(**params)

    # Train and evaluate model on training set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Compute fitness as classification accuracy
    return accuracy


# Define the Artificial Fish Swarm Algorithm
def artificial_fish_swarm(X_train, y_train, param_ranges, num_fish=50, max_iter=100, step_size=0.1, visual_range=0.1):
    # Initialize population of fish
    num_params = len(param_ranges)
    population = [{} for i in range(num_fish)]
    for i in range(num_fish):
        for p in param_ranges.keys():
            population[i][p] = np.random.choice(param_ranges[p])

    # Iterate over iterations
    best_params = None
    best_fitness = 0
    for iter in range(max_iter):
        # Compute fitness of each fish
        fitnesses = [fitness(X_train, y_train, population[i]) for i in range(num_fish)]
        if max(fitnesses) > best_fitness:
            best_params = population[np.argmax(fitnesses)]
            best_fitness = max(fitnesses)

        # Move each fish towards better food sources
        for i in range(num_fish):
            # Determine nearby fish within visual range
            neighbors = [j for j in range(num_fish) if i != j and np.linalg.norm(
                np.array(list(population[i].values())) - np.array(list(population[j].values()))) < visual_range]

            # Compute mean position of nearby fish
            mean_position = {}
            for p in param_ranges.keys():
                mean_position[p] = np.mean([population[j][p] for j in neighbors])

            # Move fish towards mean position
            for p in param_ranges.keys():
                if np.random.uniform() < 0.5:
                    population[i][p] += step_size * (mean_position[p] - population[i][p])
                else:
                    population[i][p] -= step_size * (mean_position[p] - population[i][p])

            # Ensure fish do not leave search space
            for p in param_ranges.keys():
                if population[i][p] < min(param_ranges[p]):
                    population[i][p] = min(param_ranges[p])
                elif population[i][p] > max(param_ranges[p]):
                    population[i][p] = max(param_ranges[p])

    return best_params, best_fitness


# Define the hyperparameter ranges to be tuned
param_ranges = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [10, 20, 30]
}

# Run the Artificial Fish Swarm Algorithm
best_params, best_fitness = artificial_fish_swarm(X_train, y_train, param_ranges)
best_params = {key: value for key, value in best_params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}
best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_leaves'] = int(best_params['num_leaves'])

# Create and evaluate the LightGBM classifier with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)
