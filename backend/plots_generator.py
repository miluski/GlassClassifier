import textwrap
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

class PlotsGenerator:
    
    def __init__(self, api_model):
        self.api_model = api_model
    
    def save_class_plot(self, class_counts, colors):
        plt.bar(class_counts.index, class_counts.values, color=colors)
        plt.xlabel('Klasa')
        plt.ylabel('Liczba')
        plt.title('Liczba próbek dla każdej klasy w zbiorze danych')
        plt.tight_layout()
        plt.savefig('liczba_probek_klas.png')
        plt.close()
            
    def save_feature_plot(self, feature):
        plt.legend(loc='best')
        plt.xlabel(feature)
        plt.ylabel('Częstotliwość')
        plt.title(f'Rozkład cechy {feature}')
        plt.tight_layout()
        plt.savefig(f'rozklad_cechy_{feature}.png')
        plt.close()
        self.api_model.feature_plots_paths.append(f'rozklad_cechy_{feature}.png')
    
    def save_confusion_matrix(self, title, cm, data_type, parameters_type='cv', validation_name='first_validation'):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='viridis') 
        plt.title(f'Macierz pomyłek danych {title}')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidywana etykieta')
        plt.tight_layout()
        plt.savefig(f'macierz_pomyłek_{title}.png')
        plt.close()
        if(parameters_type == 'cv'):
            self.api_model.cross_validations_object[validation_name][f'{data_type}_confusion_matrix_path'] = f'macierz_pomyłek_{title}.png'
        else:
            self.api_model.best_parameters_object[f'{data_type}_confusion_matrix_path'] = f'macierz_pomyłek_{title}.png'  
    
    def save_learning_curve_plot(self, model, X_train, y_train, title, parameters_type='cv', validation_name='first_validation'):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Dane uczące', color='red', marker='o')
        plt.plot(train_sizes, test_mean, label='Dane testowe', color='blue', marker='o')
        plt.xlabel("Ilość próbek uczących")
        plt.ylabel("Dokładność [%]")
        plt.title(f'{title}')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.grid(ls='--')
        plt.legend(loc='best')
        plt.savefig(f'{title}.png', dpi=300)
        plt.close()
        if(parameters_type == 'cv'):
            self.api_model.cross_validations_object[validation_name]['learning_curve_path'] = f'{title}.png'
        else:
            self.api_model.best_parameters_object['learning_curve_path'] = f'{title}.png'
    
    def save_metrics_plot(self, accuracy, precision, recall, specificity, npv, title, data_type, parameters_type='cv', validation_name='first_validation'):
        metrics = [accuracy, precision, recall, specificity, npv]
        metric_names = ['Dokładność', 'Precyzja', 'Czułość', 'Swoistość', 'Odsetek wyników prawdziwie negatywnych']
        wrapped_metric_names = [textwrap.fill(name, 10) for name in metric_names] 
        colors = plt.get_cmap('tab10').colors[:5]
        plt.bar(wrapped_metric_names, metrics, color=colors)
        plt.xlabel('Metryki')
        plt.ylabel('Wartości (%)')
        plt.title(f'Metryki dla danych {title}')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f'metryki_danych_{title}.png')
        plt.close()
        if(parameters_type == 'cv'):
            self.api_model.cross_validations_object[validation_name][f'{data_type}_metrics_path'] = f'metryki_danych_{title}.png'
        else:
            self.api_model.best_parameters_object[f'{data_type}_metrics_path'] = f'metryki_danych_{title}.png'   
    
    def save_loss_curve_plot(self, model, X_train, y_train, X_test, y_test, best_params, set_title, parameters_type='cv', validation_name='first_validation'):
        test_loss = []
        train_loss = []
        mlp = MLPClassifier(**best_params, warm_start=True)
        for i in range(1, len(model.loss_curve_) + 1):
            mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
            train_loss.append(mlp.loss_)
            y_test_proba = mlp.predict_proba(X_test)
            test_loss.append(log_loss(y_test, y_test_proba))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Dane uczące', color='red')
        plt.plot(range(1, len(test_loss) + 1), test_loss, label='Dane testowe', color='blue')
        plt.xlabel('Numer epoki')
        plt.ylabel('Wartość straty')
        plt.title(f'Krzywa błędu uczenia testowania {set_title}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'krzywa_bledu_uczenia_testowania_{set_title}.png')
        plt.close()
        loss_curve_path = f'krzywa_bledu_uczenia_testowania_{set_title}.png'
        if parameters_type == 'cv':
            self.api_model.cross_validations_object[validation_name]['loss_curve_path'] = loss_curve_path
        else:
            self.api_model.best_parameters_object['loss_curve_path'] = loss_curve_path