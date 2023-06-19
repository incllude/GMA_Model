from sklearn.model_selection import train_test_split
from transformations import to_bool, drop_column
from catboost.core import CatBoostClassifier
from sklearn.metrics import f1_score
from bagging import Bagging
from copy import deepcopy
import numpy as np
import optuna


def harmonic_mean(bagging_proba, cat_proba, beta):

    return (1 + beta ** 2) * bagging_proba * cat_proba / \
        (beta ** 2 * bagging_proba + cat_proba)


class Compose:

    def __init__(self, tendency=35, n_estimators=10, subsample=0.15, beta=0.6):

        self.n_estimators: int = n_estimators
        self.subsample: float = subsample
        self.tendency: float = tendency
        self.best_params: dict = None
        self.beta: float = beta

        self.bagging_model = Bagging(
            n_estimators=n_estimators,
            subsample=subsample,
            tendency=tendency
        )
        self.cat_model = CatBoostClassifier(
            verbose=False
        )
        self.big_chance_model = CatBoostClassifier(
            verbose=False
        )

    def transform(self, df_, drop_id=False):

        transform = {
            'LeftFoot': 'left',
            'RightFoot': 'right',
            'Head': 'head'
        }

        df = deepcopy(df_)
        df.drop('x', axis=1, inplace=True)
        df.drop('y', axis=1, inplace=True)
        dropping = ['match_id', 'team_id', 'team_name', 'ft_score']
        if drop_id:
            dropping.append('playerId')
        df.drop(columns=dropping, inplace=True)

        df['shootedBy'] = df['shootedBy'].apply(lambda x: transform[x])
        df['by_head'] = (df['shootedBy'] == 'head').astype(int)
        df['by_working'] = (df['shootedBy'] == df['footed']).astype(int)
        df.drop(columns=['shootedBy', 'footed'], inplace=True)
        df['bigChance'] = df['bigChance'].astype(int)

        return df

    def filter(self, df_, percent):

        df = deepcopy(df_)
        frequencies = df['playerId'].value_counts(normalize=True) * 100
        lil_ids = frequencies[frequencies > percent].index.tolist()
        df = df[df['playerId'].isin(lil_ids)]

        return df

    def clear(self, df_):

        df = deepcopy(df_)
        mean_on_train = df['distance'].mean()
        std_on_train = df['distance'].std()
        df = df[df['distance'] < mean_on_train + 2 * std_on_train]

        return df

    def fit_on_big_chance(self, x, big_chance_column):

        y = x[big_chance_column]
        x.drop(big_chance_column, axis=1, inplace=True)
        self.big_chance_model.fit(x, y)

    def split_for_few(self, x_, y_):

        x = deepcopy(x_)
        y = deepcopy(y_)
        x_train, x = train_test_split(x.assign(y=y), stratify=to_bool(y), test_size=0.66, random_state=50)
        x, y = drop_column(x, 'y')
        x_train, _ = drop_column(x_train, 'y')

        return x_train, x, y

    def edit_x(self, x_, big_chance_column):

        x = deepcopy(x_)
        if big_chance_column in x.columns:
            x, _ = drop_column(x, big_chance_column)
        x[big_chance_column] = self.big_chance_model.predict(x)

        return x

    def fit(self, x_, big_chance_column='bigChance'):

        x, y = drop_column(self.filter(self.clear(self.transform(x_)), 0.05), 'isGoal')
        x, _ = drop_column(x, 'playerId')
        x_train, x, y = self.split_for_few(x, y)
        self.fit_on_big_chance(x_train, big_chance_column)
        x = self.edit_x(x, big_chance_column)
        self.bagging_model.fit(x, y)
        self.cat_model.fit(x, to_bool(y))

    def fit_optimal(self, x_, big_chance_column='bigChance'):

        optuna.logging.set_verbosity(optuna.logging.INFO)
        x, y = drop_column(self.filter(self.clear(self.transform(x_)), 0.05), 'isGoal')
        x, _ = drop_column(x, 'playerId')
        x_train, x, y = self.split_for_few(x, y)
        self.fit_on_big_chance(x_train, big_chance_column)
        x = self.edit_x(x, big_chance_column)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, stratify=to_bool(y), test_size=0.25,
                                                              random_state=50)
        self.cat_model.fit(x_train, to_bool(y_train))

        def calculating(trial):
            beta = trial.suggest_float('beta', 0.3, 3.0, step=0.1)
            n_estimators = trial.suggest_int('n_estimators', 2, 10, step=1)
            tendency = trial.suggest_float('tendency', n_estimators, 50.0, step=1.0)
            subsample = trial.suggest_float('subsample', 0.1, 0.6, step=0.1)

            self.bagging_model = Bagging(
                n_estimators=n_estimators,
                subsample=subsample,
                tendency=tendency
            )
            self.bagging_model.fit(x_train, y_train)
            compose_predict = self.predict(x_valid, beta=beta)
            return f1_score(to_bool(y_valid), compose_predict)

        optima = optuna.create_study(direction='maximize')
        optima.optimize(calculating, n_trials=1000)
        self.best_params = optima.best_params

        self.bagging_model = Bagging(
            n_estimators=self.best_params['n_estimators'],
            subsample=self.best_params['subsample'],
            tendency=self.best_params['tendency']
        )
        self.bagging_model.fit(x_train, y_train)
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def predict_proba(self, x_, beta=None, big_chance_column='bigChance'):

        beta_selected = None
        if beta is None:
            if not self.beta is None:
                beta_selected = self.beta
            elif not self.best_params is None:
                beta_selected = self.best_params['beta']
            else:
                beta_selected = 1.0
        else:
            beta_selected = beta

        x = self.transform(x_, drop_id=True)
        x = self.edit_x(x, big_chance_column)
        bagging_proba = self.bagging_model.predict_proba(x)[:, 1]
        cat_proba = self.cat_model.predict_proba(x)[:, 1]
        compose_proba = harmonic_mean(
            bagging_proba=bagging_proba,
            cat_proba=cat_proba,
            beta=beta_selected
        )

        compose_proba = np.vectorize(lambda x: x / 1.3 if x <= 0.5 else x)(compose_proba)
        return np.transpose(np.vstack([1 - compose_proba, compose_proba]))

    def predict(self, x, beta=None, big_chance_column='bigChance'):

        beta_selected = None
        if beta is None:
            if not self.beta is None:
                beta_selected = self.beta
            elif not self.best_params is None:
                beta_selected = self.best_params['beta']
            else:
                beta_selected = 1.0
        else:
            beta_selected = beta
        compose_proba = self.predict_proba(x, beta=beta_selected, big_chance_column=big_chance_column)

        return to_bool(compose_proba[:, 1])
