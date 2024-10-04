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


# Define the energy function (or cost function)
def energy_function(X_train, y_train, params):
    # Create LightGBM classifier with specified hyperparameters
    params = {key: value for key, value in params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}
    params['max_depth'] = int(params['max_depth'])
    params['num_leaves'] = int(params['num_leaves'])
    params['learning_rate'] = abs(params['learning_rate'])
    clf = lgb.LGBMClassifier(**params)

    # Train and evaluate model on training set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Return the negative accuracy (as we want to minimize the energy function)
    return -accuracy


# Define the neighborhood function
def neighborhood_function(params, scale):
    # Generate a random perturbation vector
    perturbation = np.random.normal(scale=scale, size=len(params))

    # Apply the perturbation to the current parameters
    new_params = {k: v + perturbation[i] for i, (k, v) in enumerate(params.items())}

    return new_params


# Define the Simulated Annealing algorithm
def simulated_annealing(X_train, y_train, initial_params, energy_function, neighborhood_function, max_iter=1000, T=1.0, alpha=0.99, scale=0.1):
    current_params = initial_params.copy()
    current_energy = energy_function(X_train, y_train, current_params)
    best_params = current_params.copy()
    best_energy = current_energy
    for i in range(max_iter):
        # Generate a new solution by applying the neighborhood function
        new_params = neighborhood_function(current_params, scale)

        # Calculate the energy of the new solution
        new_energy = energy_function(X_train, y_train, new_params)

        # Calculate the acceptance probability
        delta = new_energy - current_energy
        acceptance_prob = np.exp(-delta / T)

        # Accept or reject the new solution based on the acceptance probability
        if delta < 0 or np.random.rand() < acceptance_prob:
            current_params = new_params.copy()
            current_energy = new_energy
            if current_energy < best_energy:
                best_params = current_params.copy()
                best_energy = current_energy

        # Decrease the temperature
        T *= alpha

    return best_params, -best_energy

# Define the hyperparameter ranges to be tuned
param_ranges = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [10, 20, 30]
}

# Choose initial parameters
initial_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'num_leaves': 10
}

# Run the Simulated Annealing algorithm
best_params, best_energy = simulated_annealing(X_train, y_train, initial_params, energy_function, neighborhood_function)
best_params = {key: value for key, value in best_params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}
best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['num_leaves'] = abs(best_params['num_leaves'])

# Create and evaluate the LightGBM classifier with the best hyperparameters
clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)
