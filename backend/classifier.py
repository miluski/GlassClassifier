import textwrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model import Model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

class Classifier:
    
    def __init__(self):
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
        self.cv_scores = []
        self.api_model = Model()
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.best_params:
                self.best_params[key] = value
            else:
                raise ValueError(f"Nieprawidłowy parametr: {key}")

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
        self.save_class_plot(class_counts, colors)

    def generate_feature_plots(self):
        self.features.pop()
        colors = plt.get_cmap('tab10').colors[:len(self.features)] 
        for feat in range(self.X.shape[1]):
            skew = pd.Series(self.X[:, feat]).skew()
            sns.histplot(self.X[:, feat], kde=False, color=colors[feat], label='Skośność = %.3f' % (skew), bins=30)
            self.save_feature_plot(self.features[feat])
            
    def save_class_plot(self, class_counts, colors):
        plt.bar(class_counts.index, class_counts.values, color=colors)
        plt.xlabel('Klasa')
        plt.ylabel('Liczba')
        plt.title('Liczba próbek dla każdej klasy w zbiorze danych')
        plt.savefig('liczba_probek_klas.png')
        plt.close()
            
    def save_feature_plot(self, feature):
        plt.legend(loc='best')
        plt.xlabel(feature)
        plt.ylabel('Częstotliwość')
        plt.title(f'Rozkład cechy {feature}')
        plt.savefig(f'rozklad_cechy_{feature}.png')
        plt.close()
        self.api_model.feature_plots_paths.append(f'rozklad_cechy_{feature}.png')
        
    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.train_model()

    def train_model(self):
        model = MLPClassifier(**self.best_params, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.calculate_train_scores(model, self.X_train, self.y_train, parameters_type='best')
        self.calculate_test_scores(model, self.X_test, self.y_test, parameters_type='best')
        
        self.y_train_pred = model.predict(self.X_train)
        y_pred = model.predict(self.X_test)
        cm_train = confusion_matrix(self.y_train, self.y_train_pred)
        cm = confusion_matrix(self.y_test, y_pred)   
        train_sizes, train_scores, test_scores = learning_curve(model, self.X_train, self.y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
        self.plot_learning_curve(train_sizes, train_scores, test_scores, title='Krzywa błędu uczenia dla najlepszych parametrów')
        self.plot_loss_curve(model, self.X_train, self.y_train, self.X_test, self.y_test, 'najlepszych parametrów')
        self.plot_confusion_matrix(model, self.X_train, self.y_train, 'danych uczących dla najlepszych parametrów', cm_train)
        self.plot_confusion_matrix(model, self.X_test, self.y_test, 'danych testowych dla najlepszych parametrów', cm)
        self.calculate_and_print_metrics(model, self.X_train, self.y_train, 'uczących dla najlepszych parametrów')
        self.calculate_and_print_metrics(model, self.X_test, self.y_test, 'testowych dla najlepszych parametrów')
        self.visualise_model(self.X_train, self.y_train)

    def visualise_model(self, X, y):
        skf = StratifiedKFold(n_splits=5)
        fold = 1
        for train_index, test_index in skf.split(X, y):
            self.X_train_cv, self.X_test_cv = X[train_index], X[test_index]
            self.y_train_cv, self.y_test_cv = y[train_index], y[test_index]
            model = MLPClassifier(**self.best_params, random_state=42)
            model.fit(self.X_train_cv, self.y_train_cv)
            y_pred = model.predict(self.X_test_cv)
            validation_name = self.get_validation_name(fold)
            self.calculate_train_scores(model, self.X_train_cv, self.y_train_cv, validation_name=validation_name)
            self.calculate_test_scores(model, self.y_test_cv, y_pred, validation_name=validation_name)
            
            self.y_train_pred_cv = model.predict(self.X_train_cv)
            cm_train = confusion_matrix(self.y_train_cv, self.y_train_pred_cv)
            cm = confusion_matrix(self.y_test_cv, y_pred)            
            train_sizes, train_scores, test_scores = learning_curve(model, self.X_train_cv, self.y_train_cv, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
            self.plot_learning_curve(train_sizes, train_scores, test_scores, title=f'Krzywa błędu uczenia dla walidacji numer {fold}')
            self.plot_loss_curve(model, self.X_train_cv, self.y_train_cv, self.X_test_cv, self.y_test_cv, f'dla walidacji numer {fold}')
            self.plot_confusion_matrix(model, self.X_train_cv, self.y_train_cv, f'danych uczących dla walidacji numer {fold}', cm_train)
            self.plot_confusion_matrix(model, self.X_test_cv, self.y_test_cv, f'danych testowych dla walidacji numer {fold}', cm)
            self.calculate_and_print_metrics(model, self.X_train_cv, self.y_train_cv, f'uczących dla walidacji numer {fold}')
            self.calculate_and_print_metrics(model, self.X_test_cv, self.y_test_cv, f'testowych dla walidacji numer {fold}')
            fold += 1
        avg_cv_score = np.mean(self.cv_scores)
        avg_cv_score_message = f"Średnia dokładność pięciokrotnej walidacji krzyżowej: {avg_cv_score:.2f}%"
        print(avg_cv_score_message)
    
    def get_validation_name(self, fold):
        if (fold == 1):
            return 'first_validation'
        elif (fold == 2):
            return 'second_validation'
        elif (fold == 2):
            return 'third_validation'
        elif (fold == 4):
            return 'fourth_validation'
        else:
            return 'fifth_validation'
            
    def calculate_train_scores(self, model, X_train, y_train, validation_name=None, parameters_type='cv'):
        if parameters_type == 'best':
            self.api_model.best_parameters_scores['train'] = round(model.score(X_train, y_train) * 100, 2)
        else:
            self.api_model.cross_validations_scores[validation_name]['train'] = round(model.score(X_train, y_train) * 100, 2)
        print(f'Dokladnosc uczenia {parameters_type}', round(model.score(X_train, y_train) * 100, 2))
        
    def calculate_test_scores(self, model, X_test, y_test, validation_name=None, parameters_type='cv'):
        test_score = None
        if parameters_type == 'best':
            test_score = self.api_model.best_parameters_scores['test'] = round(model.score(X_test, y_test) * 100, 2)
        else:
            test_score = test_accuracy = round(accuracy_score(X_test, y_test) * 100, 2) 
            self.api_model.cross_validations_scores[validation_name]['test'] = test_accuracy
            self.cv_scores.append(test_accuracy)
        print(f'Dokladnosc testowania {parameters_type}', test_score)
        
    def plot_learning_curve(self, train_sizes, train_scores, test_scores, title):
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='Dane Uczące', color='red', marker='o')
        plt.plot(train_sizes, test_mean, label='Dane Testowe', color='blue', marker='o')
        plt.title(title)
        plt.xlabel('Liczba próbek uczących')
        plt.ylabel('Dokładność [%]')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
        plt.grid(ls='--')
        plt.legend(loc='best')
        image_path = f'{title}.png'
        plt.savefig(image_path)
        plt.close()
        
    def plot_loss_curve(self, model, X_train, y_train, X_test, y_test, set_title):
        train_loss = []
        test_loss = []
        classes = np.unique(y_train)
        max_epochs = 2000
        best_train_epoch = 0
        best_test_epoch = 0
        lowest_train_loss = float('inf')
        lowest_test_loss = float('inf')
        for i in range(max_epochs):
            model.partial_fit(X_train, y_train, classes=classes)
            current_train_loss = 1 - model.score(X_train, y_train)
            current_test_loss = 1 - model.score(X_test, y_test)
            train_loss.append(current_train_loss)
            test_loss.append(current_test_loss)
            if current_train_loss < lowest_train_loss:
                lowest_train_loss = current_train_loss
                best_train_epoch = i
            if current_test_loss < lowest_test_loss:
                lowest_test_loss = current_test_loss
                best_test_epoch = i
        max_epochs = max(best_train_epoch, best_test_epoch) + 100
        train_loss = train_loss[:max_epochs]
        test_loss = test_loss[:max_epochs]
        plt.plot(train_loss, label='Dane Uczące', color='red')
        plt.plot(test_loss, label='Dane Testowe', color='blue')
        plt.xlabel('Numer epoki')
        plt.ylabel('Wartość straty')
        plt.title(f'Krzywa błędu uczenia/testowania {set_title}')
        plt.legend()
        image_path = f'krzywa_bledu_uczenia_testowania_{set_title}.png'
        plt.savefig(image_path)
        plt.close()

    def plot_confusion_matrix(self, model, X, y, dataset_type, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='viridis') 
        plt.title(f'Macierz pomyłek danych {dataset_type})')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidywana etykieta')
        image_path = f'macierz_pomyłek_{dataset_type}.png'
        plt.savefig(image_path)
        plt.close()
        self.print_decision_counts(cm, data_type=dataset_type)

    def print_decision_counts(self, cm, data_type):
        correct_decisions = np.trace(cm)
        total_decisions = np.sum(cm)
        incorrect_decisions = total_decisions - correct_decisions
        correct_percentage = (correct_decisions / total_decisions) * 100
        incorrect_percentage = (incorrect_decisions / total_decisions) * 100
        correct_message = f'Liczba poprawnych decyzji dla {data_type}: {correct_decisions} ({correct_percentage:.2f}%)'
        incorrect_message = f'Liczba niepoprawnych decyzji dla {data_type}: {incorrect_decisions} ({incorrect_percentage:.2f}%)'
        print(correct_message)
        print(incorrect_message)

    def calculate_and_print_metrics(self, model, X, y, dataset_type):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred) * 100
        precision = precision_score(y, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y, y_pred, average='weighted', zero_division=0) * 100
        specificity = recall_score(y, y_pred, average='weighted', labels=np.unique(y), zero_division=0) * 100
        npv = precision_score(y, y_pred, average='weighted', labels=np.unique(y), zero_division=0) * 100
        self.plot_metrics(accuracy, precision, recall, specificity, npv, dataset_type)

    def plot_metrics(self, accuracy, precision, recall, specificity, npv, dataset_type):
        metrics = [accuracy, precision, recall, specificity, npv]
        metric_names = ['Dokładność', 'Precyzja', 'Czułość', 'Swoistość', 'Odsetek wyników prawdziwie negatywnych']
        wrapped_metric_names = [textwrap.fill(name, 10) for name in metric_names] 
        colors = plt.get_cmap('tab10').colors[:5]
        plt.bar(wrapped_metric_names, metrics, color=colors)
        plt.xlabel('Metryki')
        plt.ylabel('Wartości (%)')
        plt.title(f'Metryki dla danych {dataset_type}')
        plt.ylim(0, 100)
        image_path = f'metryki_danych_{dataset_type}.png'
        plt.savefig(image_path)
        plt.close()

classifier = Classifier()
classifier.load_dataset()