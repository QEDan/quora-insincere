import logging
import numpy as np
from sklearn.linear_model import LinearRegression


class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict_average(self, X):
        logging.info("Predicting with ensemble average, size=" + str(len(self.models)))
        predictions = list()
        for m in self.models:
            predictions.append(m.predict(X))
            logging.debug(type(predictions[-1]))
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    def predict_linear_regression(self, X_train, y_train, X_predict):
        predictions_train = [model.predict(X_train) for model in self.models]
        X = np.asarray(predictions_train)
        X = X[..., 0]
        reg = LinearRegression().fit(X.T, y_train)
        predictions_predict = [model.predict(X_predict) for model in self.models]
        prediction_lin_reg = np.sum([predictions_predict[i] * reg.coef_[i]
                                     for i in range(len(predictions_predict))], axis=0)
        return prediction_lin_reg