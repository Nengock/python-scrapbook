# Load the libraries
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

# Generating a random classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42) 

# Training a gradient boosting classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42) 
gbc.fit(X, y)

# Evaluate the classifier
score = gbc.score(X, y)
print(f'The accuracy of the classifier is {score:.2f}')

