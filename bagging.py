import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def tuned_pow(a, n):
    return a ** n if a > 0 else -1 * (-a) ** n


def transform(df, tg, rate):
    indices = np.random.randint(0, df.shape[0], int(df.shape[0] * rate))
    return df.iloc[indices], tg.iloc[indices]


class Bagging:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 6,
            subsample: float = 0.2,
            tendency: float = None
    ):

        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.subsample: float = subsample
        self.tendency:float = n_estimators if tendency is None else tendency
        self.models: list = []
        self.gammas: list = []
        # self.result = LogisticRegression(n_jobs=-1)
        # self.result = DecisionTreeClassifier()
        # self.result = LinearRegression(n_jobs=-1)
        # self.result = DecisionTreeRegressor()

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def fit_new_base_model(self, x, y):
        model = self.base_model_class(
            criterion=self.base_model_params.get('criterion', 'squared_error'),
            max_depth=self.base_model_params.get('max_depth', None),
            min_samples_split=self.base_model_params.get('min_samples_split', 2),
            min_samples_leaf=self.base_model_params.get('min_samples_leaf', 1)
        )
        model.fit(x, y)
        self.models.append(model)

    def create_predictions(self, x):
        predictions = []
        for model in self.models:
            predictions.append(self.sigmoid(model.predict(x)))

        return pd.DataFrame(np.transpose(predictions))

    def make_gammas(self, x_vl, y_vl):
        predictions = self.create_predictions(x_vl)

        self.result.fit(predictions, y_vl)

    def fit(self, x_train, y_train):
        # x_tr, x_vl, y_tr, y_vl = train_test_split(x_train, y_train, test_size=0.2, random_state=50)

        for _ in range(self.n_estimators):
            x, y = transform(x_train, y_train, self.subsample)
            self.fit_new_base_model(x, y)
        # self.make_gammas(x_vl, y_vl)

    def predict_proba(self, x) -> np.array:
        # predictions = self.create_predictions(x)
        # prediction_to_1 = self.sigmoid(self.result.predict(predictions))
        # return np.transpose([1 - prediction_to_1, prediction_to_1])

        predictions = np.empty((0, x.shape[0]))
        for model in self.models:
            predictions = np.vstack([predictions, self.sigmoid(model.predict(x))])
        predictions = np.transpose(predictions)

        geometric_mean = lambda probas: np.power(np.prod(probas), 1 / self.tendency)
        predictions = np.apply_along_axis(geometric_mean, 1, predictions)

        return np.transpose([1 - predictions, predictions])

    def predict(self, x) -> np.array:
        probas = self.predict_proba(x)
        predict = np.vectorize(lambda x: True if x > 0.5 else False)(probas[:, 1])

        return predict
