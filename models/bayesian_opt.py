import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the objective function to be optimized
def lgbm_cv(num_leaves, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {'num_leaves': int(num_leaves),
              'learning_rate': learning_rate,
              'min_child_weight': int(min_child_weight),
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'metric': 'auc'}
    lgbm = lgb.LGBMClassifier(**params, n_estimators=1000, n_jobs=-1)
    score = cross_val_score(lgbm, X_train, y_train, scoring='roc_auc', cv=5).mean()
    return score

# Define the search space for the hyperparameters
params = {'num_leaves': (24, 45),
          'learning_rate': (0.01, 0.1),
          'min_child_weight': (1, 10),
          'subsample': (0.5, 1),
          'colsample_bytree': (0.5, 1),
          'reg_alpha': (0, 1),
          'reg_lambda': (0, 1)}

# Run Bayesian Optimization
lgbmBO = BayesianOptimization(f=lgbm_cv, pbounds=params, random_state=1)
lgbmBO.maximize(init_points=5, n_iter=20)

# Get the best hyperparameters and evaluate the model on the test set
best_params = lgbmBO.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

lgbm = lgb.LGBMClassifier(**best_params, n_estimators=1000, n_jobs=-1)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_params)
print("Accuracy on test set:", accuracy)
