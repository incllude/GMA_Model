from sklearn.model_selection import train_test_split
from transformations import *
from compose import Compose
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
random_state = 50

bundes = pd.read_csv('data/shots_bundes_extended.csv').dropna(axis=0)
apl = pd.read_csv('data/shots_apl_extended.csv').dropna(axis=0)
rpl = pd.read_csv('data/shots_rpl_extended.csv').dropna(axis=0)

x_train, x_test = train_test_split(pd.concat([bundes, apl, rpl]), test_size=0.25)

compose = Compose()
compose.fit(x_train)

show_results(x_test['isGoal'], compose.predict_proba(x_test.drop('isGoal', axis=1))[:, 1])
