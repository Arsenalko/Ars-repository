# <YOUR_IMPORTS>
import dill
import os
import pandas as pd
import json

from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')


def predict():
    d = datetime.now().strftime("%Y%m%d%H%M")
    with open(f'{path}/data/models/cars_pipe_{d}.pkl', 'rb') as file:
        model = dill.load(file)

    id_list = []
    pred_list = []
    for filename in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{filename}', 'r') as j:
            df = pd.DataFrame(json.load(j), index=[0])
        id_list.append(df['id'].tolist())
        pred_list.append(model.predict(df))

    df_preds = pd.DataFrame({'id': id_list, 'predictions': pred_list})
    df_preds.loc[:, ['id', 'predictions']].to_csv(f'{path}/data/predictions/preds_{d}.csv', index="False")


if __name__ == '__main__':
    predict()
