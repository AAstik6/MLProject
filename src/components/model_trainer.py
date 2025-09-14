import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
      try:
        logging.info("split the training and test input data")
        X_train, y_train, X_test, y_test = (
          train_array[:,:-1], # input features of the training data
          train_array[:,-1], # output feature of the training data
          test_array[:,:-1], # input features of the test data
          test_array[:,-1]  # output feature of the test data
        )

        models = {
          "Random Forest": RandomForestRegressor(),
          "Decision Tree": DecisionTreeRegressor(),
          "Gradient Boosting": GradientBoostingRegressor(),
          "Linear Regression": LinearRegression(),
          "AdaBoost Regressor": AdaBoostRegressor(),
        }

        params = {
          "Random Forest": {
            'n_estimators': [8,16,32,64,128,256],
          },
          "Decision Tree": {
            'max_depth': [5, 10, None],
            'criterion': ["squared_error", "friedman_mse", "absolute_error"],
          },
          "Gradient Boosting": {
            'learning_rate': [.1, .01, .05, .001],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'n_estimators': [8,16,32,64,128,256],
          },
          "Linear Regression": {},

          "AdaBoost Regressor": {
            'learning_rate': [.1, .01, 0.5, .001],
            'n_estimators': [8,16,32,64,128,256],
          },
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models = models, params = params)

        # to get the best model score from the dict
        valid_scores = [score for score in model_report.values() if score is not None]
        best_model_score = max(valid_scores)

        # to get the best model name from the dict
        best_model_name = list(model_report.keys())[
          list(model_report.values()).index(best_model_score)
          ]
        
        best_model = models[best_model_name]

        if best_model_score<0.6:
          raise CustomException("No best model found.")
        
        logging.info(f"Best model found on both training and testing dataset")
        
        save_object(
          file_path = self.model_trainer_config.trained_model_file_path,
          obj = best_model
        )
        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)
        return r2_square
      except Exception as e:
        raise CustomException(e, sys)