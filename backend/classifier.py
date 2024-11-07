import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from api_model import ApiModel
from plots_generator import PlotsGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class Classifier:
    
    def __init__(self, **params):
        self.cv_scores = []
        self.api_model = ApiModel()
        self.plots_generator = PlotsGenerator(self.api_model)
        self.best_params = {
            'activation': 'logistic',
            'alpha': 0.01,
            'batch_size': 'auto',
            'hidden_layer_sizes': (100,),
            'learning_rate_init': 0.1,
            'max_iter': 2000,
            'solver': 'adam',
            'tol': 0.0001
        }
        self.features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        for key, value in params.items():
            if key in self.best_params:
                if key == 'hidden_layer_sizes' and isinstance(value, str):
                    value = ast.literal_eval(value)
                self.best_params[key] = value
            else:
                raise ValueError(f"Nieprawidłowy parametr: {key}")
        self.load_dataset()

    def load_dataset(self):
        self.data = pd.read_csv('glass.csv', names=self.features)
        self.X = self.data.drop(columns=['Type']).values  
        self.y = self.data['Type'].values 
        self.generate_class_plot()
        self.generate_feature_plots()
        self.split_dataset()

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=1)
        self.normalize_data()

    def generate_class_plot(self):
        class_counts = pd.Series(self.y).value_counts()
        colors = plt.get_cmap('tab10').colors[:len(class_counts)]
        self.plots_generator.save_class_plot(class_counts, colors)

    def generate_feature_plots(self):
        self.features.pop()
        colors = plt.get_cmap('tab10').colors[:len(self.features)] 
        for feat in range(self.X.shape[1]):
            skew = pd.Series(self.X[:, feat]).skew()
            sns.histplot(self.X[:, feat], kde=False, color=colors[feat], label='Skośność = %.3f' % (skew), bins=30)
            self.plots_generator.save_feature_plot(self.features[feat])
            
    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.train_model()

    def train_model(self):
        model = MLPClassifier(**self.best_params, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.calculate_train_scores(model, parameters_type='best')
        self.calculate_test_scores(model, parameters_type='best')
        self.generate_curves(model, parameters_type='best')
        self.generate_training_data_metrics(model, parameters_type='best')
        self.generate_test_data_metrics(model, parameters_type='best')
        self.validate_model()
    
    def validate_model(self):
        skf = StratifiedKFold(n_splits=5)
        self.fold = 1
        for train_index, test_index in skf.split(self.X_train, self.y_train):
            model = MLPClassifier(**self.best_params, random_state=42)
            self.X_train_cv, self.X_test_cv = self.X_train[train_index], self.X_train[test_index]
            self.y_train_cv, self.y_test_cv = self.y_train[train_index], self.y_train[test_index]
            model.fit(self.X_train_cv, self.y_train_cv)
            self.y_pred_cv = model.predict(self.X_test_cv)
            validation_name = self.get_validation_name(self.fold)
            self.calculate_train_scores(model, validation_name)
            self.calculate_test_scores(model, validation_name)
            self.generate_curves(model)
            self.generate_training_data_metrics(model)
            self.generate_test_data_metrics(model)          
            self.fold += 1
        self.api_model.average_cross_validation_score = np.mean(self.cv_scores)
    
    def calculate_train_scores(self, model, validation_name=None, parameters_type='cv'):
        if parameters_type == 'best':
            self.api_model.best_parameters_object['train'] = round(model.score(self.X_train, self.y_train) * 100, 2)
        else:
            self.api_model.cross_validations_object[validation_name]['train'] = round(model.score(self.X_train_cv, self.y_train_cv) * 100, 2)
        
    def calculate_test_scores(self, model, validation_name=None, parameters_type='cv'):
        if parameters_type == 'best':
            self.api_model.best_parameters_object['test'] = round(model.score(self.X_test, self.y_test) * 100, 2)
        else:
            test_accuracy = round(accuracy_score(self.y_test_cv, self.y_pred_cv) * 100, 2) 
            self.api_model.cross_validations_object[validation_name]['test'] = test_accuracy
            self.cv_scores.append(test_accuracy)

    def generate_curves(self, model, parameters_type='cv'):
        if (parameters_type=='best'):
            self.plots_generator.save_learning_curve_plot(model, self.X_train, self.y_train, title='Krzywa uczenia dla najlepszych parametrów', parameters_type=parameters_type)
            self.plots_generator.save_loss_curve_plot(model, self.X_train, self.y_train, self.X_test, self.y_test, self.best_params, 'najlepszych parametrów', parameters_type)
        else:
            validation_name = self.get_validation_name(self.fold)
            self.plots_generator.save_learning_curve_plot(model, self.X_train_cv, self.y_train_cv, title=f'Krzywa uczenia dla walidacji numer {self.fold}', validation_name=validation_name)
            self.plots_generator.save_loss_curve_plot(model, self.X_train_cv, self.y_train_cv, self.X_test_cv, self.y_test_cv, self.best_params, f'walidacji numer {self.fold}', parameters_type, validation_name)
            
    def generate_training_data_metrics(self, model, parameters_type='cv'):
        if (parameters_type=='best'):
            cm = confusion_matrix(self.y_train, model.predict(self.X_train))
            self.plots_generator.save_confusion_matrix('danych uczących dla najlepszych parametrów', cm, 'train', parameters_type=parameters_type)
            self.calculate_metrics(model, self.X_train, self.y_train, 'uczących dla najlepszych parametrów', 'train', parameters_type)
        else:
            validation_name = self.get_validation_name(self.fold)
            cm = confusion_matrix(self.y_train_cv, model.predict(self.X_train_cv))
            self.plots_generator.save_confusion_matrix(f'danych uczących dla walidacji numer {self.fold}', cm, 'train', parameters_type=parameters_type, validation_name=validation_name)
            self.calculate_metrics(model, self.X_train_cv, self.y_train_cv, f'uczących dla walidacji numer {self.fold}', 'train', parameters_type)
        self.calculate_decisions_count(cm, data_type='train_data', parameters_type=parameters_type)

    def generate_test_data_metrics(self, model, parameters_type='cv'):
        if (parameters_type=='best'):
            cm = confusion_matrix(self.y_test, model.predict(self.X_test))
            self.plots_generator.save_confusion_matrix('danych testowych dla najlepszych parametrów', cm, 'test', parameters_type=parameters_type)
            self.calculate_metrics(model, self.X_test, self.y_test, 'testowych dla najlepszych parametrów', 'test', parameters_type)
        else:       
            validation_name = self.get_validation_name(self.fold)         
            cm = confusion_matrix(self.y_test_cv, self.y_pred_cv)
            self.plots_generator.save_confusion_matrix(f'danych testowych dla walidacji numer {self.fold}', cm, 'test', parameters_type=parameters_type, validation_name=validation_name)
            self.calculate_metrics(model, self.X_test_cv, self.y_test_cv, f'testowych dla walidacji numer {self.fold}', 'test')
        self.calculate_decisions_count(cm, data_type='test_data', parameters_type=parameters_type)
        
    def calculate_decisions_count(self, cm, data_type, parameters_type='cv'):
        correct_decisions = np.trace(cm)
        total_decisions = np.sum(cm)
        if(parameters_type == 'cv'):
            validation_name = self.get_validation_name(self.fold)
            self.api_model.cross_validations_object[validation_name]['correct_decisions'][data_type] = correct_decisions
            self.api_model.cross_validations_object[validation_name]['incorrect_decisions'][data_type] = total_decisions - correct_decisions
            self.api_model.cross_validations_object[validation_name]['correct_percentage'][data_type] = (correct_decisions / total_decisions) * 100
            self.api_model.cross_validations_object[validation_name]['incorrect_percentage'][data_type] = ((total_decisions - correct_decisions) / total_decisions) * 100
        else:
            self.api_model.best_parameters_object['correct_decisions'][data_type] = correct_decisions
            self.api_model.best_parameters_object['incorrect_decisions'][data_type] = total_decisions - correct_decisions
            self.api_model.best_parameters_object['correct_percentage'][data_type] = (correct_decisions / total_decisions) * 100
            self.api_model.best_parameters_object['incorrect_percentage'][data_type] = ((total_decisions - correct_decisions) / total_decisions) * 100
            
    def calculate_metrics(self, model, X, y, title, data_type, parameters_type='cv'):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred) * 100
        precision = precision_score(y, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y, y_pred, average='weighted', zero_division=0) * 100
        specificity = recall_score(y, y_pred, average='weighted', labels=np.unique(y), zero_division=0) * 100
        npv = precision_score(y, y_pred, average='weighted', labels=np.unique(y), zero_division=0) * 100
        if(parameters_type == 'cv'):
            validation_name = self.get_validation_name(self.fold)
            self.api_model.cross_validations_object[validation_name]['accuracy'] = accuracy
            self.api_model.cross_validations_object[validation_name]['precision'] = precision
            self.api_model.cross_validations_object[validation_name]['recall'] = recall
            self.api_model.cross_validations_object[validation_name]['specificity'] = specificity
            self.api_model.cross_validations_object[validation_name]['npv'] = npv   
            self.plots_generator.save_metrics_plot(accuracy, precision, recall, specificity, npv, title, data_type, parameters_type, validation_name)
        else:
            self.api_model.best_parameters_object['accuracy'] = accuracy
            self.api_model.best_parameters_object['precision'] = precision
            self.api_model.best_parameters_object['recall'] = recall
            self.api_model.best_parameters_object['specificity'] = specificity
            self.api_model.best_parameters_object['npv'] = npv
            self.plots_generator.save_metrics_plot(accuracy, precision, recall, specificity, npv, title, data_type, parameters_type)
    
    def get_validation_name(self, fold):
        if (fold == 1):
            return 'first_validation'
        elif (fold == 2):
            return 'second_validation'
        elif (fold == 3):
            return 'third_validation'
        elif (fold == 4):
            return 'fourth_validation'
        else:
            return 'fifth_validation'