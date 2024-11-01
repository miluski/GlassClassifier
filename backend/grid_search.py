import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class GridSearchClassifier:
    
    def load_dataset(self):
        column_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        data = pd.read_csv('glass.csv', names=column_names)
        X = data.drop(columns=['Type'])
        y = data['Type']
        self.split_dataset(X, y)

    def split_dataset(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        self.normalize_data(X_train, X_test, y_train, y_test)

    def normalize_data(self, X_train, X_test, y_train, y_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.grid_search(X_train, X_test, y_train, y_test)

    def grid_search(self, X_train, X_test, y_train, y_test):
        param_grid = {
            'hidden_layer_sizes': [(5, 2), (10, 5), (10, 10), (50, 50), (100,)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [2000, 5000, 10000],
            'tol': [1e-4, 1e-5, 1e-6],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'batch_size': ['auto', 32, 64, 128]
        }
        model = MLPClassifier(random_state=1)
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best cross-validation score: {grid_search.best_score_}')
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        print(f'Training score: {best_model.score(X_train, y_train)}')
        print(f'Test score: {best_model.score(X_test, y_test)}')

GridSearchClassifier().load_dataset()