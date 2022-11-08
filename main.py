import math
from itertools import islice
import seaborn as sns
import statistics as st
import numpy as np
from typing import List, Any, Union

from scapy.all import *
import matplotlib.pyplot as plt
import pandas as pd

ANOMALY_THRESHOLD = 0.99
MILD_ANOMALY_THRESHOLD = 0.95
SLIDING_WINDOW = 200
SLIDING_WINDOW_LARGE = 150


def logistic(b, x, x0):
    if -b * (x - x0) > 100:
        logic = 0.9999999999999999
    else:
        logic = 1 / (1 + math.e ** (-b * (x - x0)))
    return logic


def lastNaverage(lst, N):
    res = list(islice(reversed(lst), 0, N))
    res.reverse()
    if len(res) == 0:
        return 0
    else:
        return sum(res) / len(res)


# SARMA/SARIMA prediction models
def predictNextNvalues(history, N):
    ###
    return [17.4, 17.2, 16.8, 18.4]


def anomalydetection(timeseries):
    predicted = timeseries[0]
    predictions = []
    windows = []
    deltas = []
    anomaly = []
    likelihood = 0
    variance = 0
    variances = []
    WIN_AVG = SLIDING_WINDOW

    for point in range(0, len(timeseries)):
        # calculating absolute value of deviation of prediction from actual point
        delta = abs(timeseries[point] - predicted)
        deltas.append(delta)
        predictions.append(predicted)
        windows.append(WIN_AVG)
        variances.append(variance)

        if lastNaverage(deltas, SLIDING_WINDOW) == 0:
            likelihood = 0
            variance = 0.0
        else:
            variance = logistic(6, delta / lastNaverage(deltas, SLIDING_WINDOW), 2.3)
            likelihood = logistic(6, likelihood + variance, 0.9)

        anomaly.append(likelihood)
        WIN_AVG = int(SLIDING_WINDOW * (1 + logistic(10, likelihood, 0.1)))

        alpha = logistic(8, lastNaverage(anomaly, int(lastNaverage(windows, WIN_AVG))), 0.2)
        predicted = (alpha) * timeseries[point] + (1 - alpha) * predicted

    return anomaly, windows, predictions, variances


def get_anomaly(timeseries, name):
    anomalydots = []
    mildanomalydots = []
    xdots = []
    xmilddots = []
    timerange = []
    y = []

    for i in range(0, len(timeseries)):
        timerange.append(i)
        y.append(0.9)

    anomaly, windows, predictions, variances = anomalydetection(timeseries)

    # defining anomaly dots for both X and Anomaly indicator series
    for i in range(0, len(timeseries)):
        if anomaly[i] >= MILD_ANOMALY_THRESHOLD:
            if anomaly[i] >= ANOMALY_THRESHOLD:
                xdots.append(timeseries[i - 1])
                anomalydots.append(anomaly[i])
                xmilddots.append(None)
                mildanomalydots.append(None)
            else:
                xmilddots.append(timeseries[i - 1])
                mildanomalydots.append(anomaly[i])
                xdots.append(None)
                anomalydots.append(None)
        else:
            xdots.append(None)
            xmilddots.append(None)
            anomalydots.append(None)
            mildanomalydots.append(None)

    xdots.pop(0)
    xdots.append(None)
    xmilddots.pop(0)
    xmilddots.append(None)

    fig, (sp1, sp2, sp3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(name)

    sp1.spines['top'].set_visible(False)
    sp1.spines['right'].set_visible(False)
    sp1.spines['left'].set_visible(False)
    sp1.grid(axis='y', color='silver', linestyle='--', linewidth=0.8)

    sp2.spines['top'].set_visible(False)
    sp2.spines['right'].set_visible(False)
    sp2.spines['left'].set_visible(False)
    sp2.grid(axis='y', color='silver', linestyle='--', linewidth=0.8)

    sp3.spines['top'].set_visible(False)
    sp3.spines['right'].set_visible(False)
    sp3.spines['left'].set_visible(False)
    sp3.grid(axis='y', color='silver', linestyle='--', linewidth=0.8)

    sp1.title.set_text('Initial timeseries: lines')
    sp2.title.set_text('Initial timeseries: dots')
    sp3.title.set_text('Anomaly likelihood')
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.25)

    sp1.plot(timerange, timeseries, '-', color="silver", alpha=0.7, linewidth=0.5)
    sp1.plot(timerange, xdots, '.', color="maroon", linewidth=3)
    sp1.plot(timerange, xmilddots, '.', color="darkred", alpha=0.3, linewidth=3)

    sp2.plot(timerange, timeseries, '.', color="silver", alpha=0.4, linewidth=0.15)
    sp2.plot(timerange, xdots, '.', color="maroon", linewidth=2)
    sp2.plot(timerange, xmilddots, '.', color="darkred", alpha=0.3, linewidth=3)

    sp3.set_ylim(0, 1.0)
    sp3.plot(timerange, anomaly, '-', color="silver", alpha=0.7)
    # sp3.plot(timerange, windows, '-', color="blue", alpha=0.7)
    # sp3.plot(timerange, variances, '-', color="gray", alpha=0.7)
    # sp3.plot(timerange, y, '-', alpha=1)
    # sp3.plot(timerange, anomalydots, 'r.')
    # sp3.plot(timerange, mildanomalydots, '.', color="orange")

    # plt.hist(timeseries, bins=int(180 / 1), color='blue', edgecolor='black')

    # sp2.fill_between(timerange, list(map(lambda x, y: x - y, predictions, boundary)), list(map(lambda x, y: x + y, predictions, boundary)), facecolor='lightblue')

    plt.savefig('data/output/col0/' + name + '.png', dpi=600)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    timeseries = []
    data = pd.read_csv('data/timeseries1.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries2.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries3.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries4.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries5.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries6.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries7.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries8.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries9.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries10.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries11.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries12.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries13.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries14.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries15.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries16.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries17.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries18.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries19.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries20.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/timeseries21.csv')
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/MED-LAN-2GHz4-100.csv', sep=';', usecols=[0], names=['value'], header=None)
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/MED-LAN-5GHz-100.csv', sep=';', usecols=[0], names=['value'], header=None)
    timeseries.append(data['value'].values)
    data = pd.read_csv('data/MED-LAN-CAT6-100.csv', sep=';', usecols=[0], names=['value'], header=None)
    timeseries.append(data['value'].values)

    # get_anomaly(timeseries[0], 'Timeseries 0: Simple anomaly')

    # get_anomaly(timeseries[1], 'Timeseries 1: ')

    # get_anomaly(timeseries[2], 'Timeseries 2')
    # get_anomaly(timeseries[2][0:580], 'Timeseries 2: points 0..580')

    # get_anomaly(timeseries[3], 'Timeseries 3')

    # get_anomaly(timeseries[4], 'Timeseries 4')

    # get_anomaly(timeseries[5], 'Timeseries 5')
    # sns.displot(timeseries[5])

    # get_anomaly(timeseries[6], 'Timeseries 6')
    # get_anomaly(timeseries[6][:150], 'Timeseries 6: points 0..150')

    # get_anomaly(timeseries[7], 'Timeseries 7. SPLUNK Sample dataset')
    # get_anomaly(timeseries[7][:200], 'Timeseries 7 0:200')

    # get_anomaly(timeseries[8], 'Timeseries 8')

    # get_anomaly(timeseries[9], 'Timeseries 9: Numenta Anomaly Benchmark - Artificial no Anomaly no noise')

    # get_anomaly(timeseries[10], 'Timeseries 10: Numenta Anomaly Benchmark - Artificial no Anomaly small noise')

    # get_anomaly(timeseries[11], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise')
    # get_anomaly(timeseries[11][:500], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise Zoomed 0:500 points')
    # get_anomaly(timeseries[11][750:1000], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise Zoomed 750:1000 points')

    # get_anomaly(timeseries[12], 'Artificial ANOMALY down')
    # get_anomaly(timeseries[12][2500:3300], 'Artificial ANOMALY down zoomed')

    # get_anomaly(timeseries[13], 'Artificial ANOMALY flat middle')
    # get_anomaly(timeseries[13][2000:3000], 'Artificial ANOMALY flat middle')

    # get_anomaly(timeseries[14], 'Artificial ANOMALY increase spike density')

    # get_anomaly(timeseries[15], 'Timeseries 15: Real Traffic ANOMALY speed')
    # get_anomaly(timeseries[15][2000:2500], 'Timeseries 15: Real Traffic ANOMALY speed 2000..2500')

    # get_anomaly(timeseries[16], 'Timeseries 16: Real Traffic ANOMALY speed zoomed')
    # get_anomaly(timeseries[16][2450:2650], 'Timeseries 16: Real Traffic ANOMALY speed zoomed')
    # get_anomaly(timeseries[16][16000:], 'Timeseries 16: Real Traffic ANOMALY speed points 16000...')

    # get_anomaly(timeseries[17], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[17][:200], 'Real Traffic ANOMALY Travel Time :200')
    # get_anomaly(timeseries[17][200:600], 'Real Traffic ANOMALY Travel Time 200:600')
    # get_anomaly(timeseries[17][600:1000], 'Real Traffic ANOMALY Travel Time 600:1000')

    # get_anomaly(timeseries[18], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[18][:500], 'Real Traffic ANOMALY Travel Time')

    # get_anomaly(timeseries[19], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[19][1050:1300], 'Real Traffic ANOMALY Travel Time')

    # get_anomaly(timeseries[20], 'Figure 3')
    # get_anomaly(timeseries[20][:300], 'Figure 2')

    subplots_count = 20
    # groupby = SLIDING_WINDOW

    # get_anomaly(timeseries[21], 'MED-LAN-2GHz4-100 :: full range')
    # for i in range(0, subplots_count):
    #     get_anomaly(timeseries[21][int(i * len(timeseries[21]) / subplots_count): int((i + 1) * len(timeseries[21]) / subplots_count - 1)], f'MED-LAN-2GHz4-100 :: {int(i * len(timeseries[21]) / subplots_count)}:{int((i + 1) * len(timeseries[21]) / subplots_count - 1)}')
    #
    # get_anomaly(timeseries[22], 'MED-LAN-5GHz-100 :: full range')
    # for i in range(0, subplots_count):
    #     get_anomaly(timeseries[22][int(i * len(timeseries[22]) / subplots_count): int((i + 1) * len(timeseries[22]) / subplots_count - 1)], f'MED-LAN-5GHz-100 :: {int(i * len(timeseries[22]) / subplots_count)}:{int((i + 1) * len(timeseries[22]) / subplots_count - 1)}')
    #
    # get_anomaly(timeseries[23], 'MED-LAN-CAT6-100 :: full range')
    # for i in range(0, subplots_count):
    #     get_anomaly(timeseries[23][int(i * len(timeseries[23]) / subplots_count): int((i + 1) * len(timeseries[23]) / subplots_count - 1)], f'MED-LAN-CAT6-100 :: {int(i * len(timeseries[23]) / subplots_count)}:{int((i + 1) * len(timeseries[23]) / subplots_count - 1)}')

    SLIDING_WINDOW = 30
    for j in range(1, 11):
        groupby = SLIDING_WINDOW * j

        subList = [sum(timeseries[21][i: i + groupby]) / groupby for i in range(0, len(timeseries[21]), groupby)]
        get_anomaly(subList[:-1], f'MED-LAN-2GHz4-100 :: groupby SLIDING_WINDOW ({groupby})')

        # subList = [sum(timeseries[22][i: i + groupby]) / groupby for i in range(0, len(timeseries[22]), groupby)]
        # get_anomaly(subList[:-1], f'MED-LAN-5GHz-100 :: groupby SLIDING_WINDOW ({groupby})')
        #
        # subList = [sum(timeseries[23][i: i + groupby]) / groupby for i in range(0, len(timeseries[23]), groupby)]
        # get_anomaly(subList[:-1], f'MED-LAN-CAT6-100 :: groupby SLIDING_WINDOW ({groupby})')
