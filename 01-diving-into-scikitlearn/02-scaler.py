# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#Initializing the StandardScaler
scaler = StandardScaler()

# Generating a random classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# Scaling the training set
X_train_scaled = scaler.fit_transform(X_train)

# Output the mean and standard deviation of the first feature before and after scaling
print(f'The mean of the first feature before scaling is {X_train[:,0].mean():.2f}') 
print(f'The standard deviation of the first feature before scaling is {X_train[:,0].std():.2f}')    
print(f'The mean of the first feature after scaling is {X_train_scaled[:,0].mean():.2f}')
print(f'The standard deviation of the first feature after scaling is {X_train_scaled[:,0].std():.2f}')
