import pandas as pd
import csv

folder = 'data'

if __name__ == '__main__':
    real_data = pd.read_csv(folder + '/2016-2017_result.csv')
    pred_data = pd.read_csv('16-17Result.csv')
    result = []
    pred_list = []
    for index, row in pred_data.iterrows():
        pred_list.append(row)

    probability = []
    for index, row in real_data.iterrows():
        Wteam = row['WTeam']
        if pred_list[index]['win'] == Wteam:
            result.append(1)
        else:
            result.append(0)
            probability.append(pred_list[index]['probability'])

    print(result.count(1) * 1.0 / len(result))
