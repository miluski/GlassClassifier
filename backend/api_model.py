import numpy as np

class ApiModel:
    
    def __init__(self):
        self.class_ploth_path = "liczba_probek_klas.png"
        self.feature_plots_paths = []
        self.best_parameters_object = {
            'train': None,
            'test': None,
            'learning_curve_path': None,
            'loss_curve_path': None,
            'train_confusion_matrix_path': None,
            'test_confusion_matrix_path': None,
            'train_metrics_path': None,
            'test_metrics_path': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'specificity': None,
            'npv': None,
            'correct_decisions': {
                'train_data': None,
                'test_data': None
            },
            'incorrect_decisions': {
                'train_data': None,
                'test_data': None
            },
            'correct_percentage': {
                'train_data': None,
                'test_data': None
            },
            'incorrect_percentage': {
                'train_data': None,
                'test_data': None
            },
        }
        self.cross_validations_object = {
            'first_validation': {
                'train': None,
                'test': None,
                'learning_curve_path': None,
                'loss_curve_path': None,
                'train_confusion_matrix_path': None,
                'test_confusion_matrix_path': None,
                'train_metrics_path': None,
                'test_metrics_path': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'specificity': None,
                'npv': None,
                'correct_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'correct_percentage': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_percentage': {
                    'train_data': None,
                    'test_data': None
                },
            },
            'second_validation': {
                'train': None,
                'test': None,
                'learning_curve_path': None,
                'loss_curve_path': None,
                'train_confusion_matrix_path': None,
                'test_confusion_matrix_path': None, 
                'train_metrics_path': None,
                'test_metrics_path': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'specificity': None,
                'npv': None,
                'correct_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'correct_percentage': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_percentage': {
                    'train_data': None,
                    'test_data': None
                },
            },
            'third_validation': {
                'train': None,
                'test': None,
                'learning_curve_path': None,
                'loss_curve_path': None,
                'train_confusion_matrix_path': None,
                'test_confusion_matrix_path': None,
                'train_metrics_path': None,
                'test_metrics_path': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'specificity': None,
                'npv': None,
                'correct_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'correct_percentage': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_percentage': {
                    'train_data': None,
                    'test_data': None
                },
            },
            'fourth_validation': {
                'train': None,
                'test': None,
                'learning_curve_path': None,
                'loss_curve_path': None,
                'train_confusion_matrix_path': None,
                'test_confusion_matrix_path': None,
                'train_metrics_path': None,
                'test_metrics_path': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'specificity': None,
                'npv': None,              
                'correct_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'correct_percentage': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_percentage': {
                    'train_data': None,
                    'test_data': None
                },
            },
            'fifth_validation': {
                'train': None,
                'test': None,
                'learning_curve_path': None,
                'loss_curve_path': None,
                'train_confusion_matrix_path': None,
                'test_confusion_matrix_path': None,
                'train_metrics_path': None,
                'test_metrics_path': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'specificity': None,
                'npv': None,
                'correct_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_decisions': {
                    'train_data': None,
                    'test_data': None
                },
                'correct_percentage': {
                    'train_data': None,
                    'test_data': None
                },
                'incorrect_percentage': {
                    'train_data': None,
                    'test_data': None
                },
            },
        }
        self.average_cross_validation_score = None

    def to_dict(self):
        def convert_value(value):
            if isinstance(value, np.generic):
                return value.item()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            return value

        return {
            'class_ploth_path': self.class_ploth_path,
            'feature_plots_paths': self.feature_plots_paths,
            'best_parameters_object': convert_value(self.best_parameters_object),
            'cross_validations_object': convert_value(self.cross_validations_object),
            'average_cross_validation_score': self.average_cross_validation_score
        }

    def __str__(self):
        return (
            f"ApiModel(\n"
            f"  class_ploth_path={self.class_ploth_path},\n"
            f"  feature_plots_paths={self.feature_plots_paths},\n"
            f"  best_parameters_object={self.best_parameters_object},\n"
            f"  cross_validations_object={self.cross_validations_object},\n"
            f"  average_cross_validation_score={self.average_cross_validation_score}\n"
            f")"
        )