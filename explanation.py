import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lime import lime_tabular

# Load the dataset
data = load_breast_cancer()
X = data['data']
y = data['target']
feature_names = data['feature_names']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
model = lgb.train(params, lgb_train, num_boost_round=100)

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['benign', 'malignant'])

# Explain a single instance using LIME
instance_to_explain = X_test[0]
exp = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=len(feature_names))
