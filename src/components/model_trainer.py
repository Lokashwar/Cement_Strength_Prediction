import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_hyperparameters(self, model, param_grid, X_train, y_train, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1):
        logging.info("Starting hyperparameter tuning...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, 
                                   cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        logging.info(f"Hyperparameter tuning completed. Best Parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'SVR': SVR(kernel='poly', C=100, degree=3, epsilon=0.1, gamma="scale")
            }

            param_grid = {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }


            tuned_rf_model, tuned_rf_params = self.tune_hyperparameters(
            model=models['RandomForest'], 
            param_grid=param_grid['RandomForest'], 
            X_train=X_train, 
            y_train=y_train
            )

            models['RandomForest'] = tuned_rf_model

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"Best Model: {best_model_name} with a score of {best_model_score}")
            logging.info(f"Best Model: {best_model_name} with a score of {best_model_score}")

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)