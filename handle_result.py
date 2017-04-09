import pandas as pd
import csv
from datetime import datetime

folder = 'temp'

if __name__ == '__main__':
    schedule = pd.read_csv(folder + '/16-17Schedule_Result.csv')
    for index, row in schedule.iterrows():
        row['Date'] = row['Date'] + ' ' + row['Time'].split(' ')[0]
        row['Date'] = str(datetime.strptime(row['Date'], "%a %b %d %Y %I:%M"))
        schedule.ix[index] = row
    new_schedule = schedule.drop(['Time'], axis=1)
    new_schedule = new_schedule.sort_values(by='Date')
    new_schedule.to_csv(folder + '/16-17Schedule_Result1.csv',index=False)