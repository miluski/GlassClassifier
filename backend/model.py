class Model:
    def __init__(self):
        self.class_ploth_path = "liczba_probek_klas.png"
        self.feature_plots_paths = []
        self.best_parameters_scores = {
            'train': None,
            'test': None
        }
        self.cross_validations_scores = {
            'first_validation': {
                'train': None,
                'test': None
            },
            'second_validation': {
                'train': None,
                'test': None
            },
            'third_validation': {
                'train': None,
                'test': None
            },
            'fourth_validation': {
                'train': None,
                'test': None
            },
            'fifth_validation': {
                'train': None,
                'test': None
            },
        }