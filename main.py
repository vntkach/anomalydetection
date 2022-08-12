import math
from itertools import islice
import statistics as st
import numpy as np
from typing import List, Any, Union

from scapy.all import *
import matplotlib.pyplot as plt
import pandas as pd


ANOMALY_THRESHOLD = 0.98
MILD_ANOMALY_THRESHOLD = 0.9
SLIDING_WINDOW = 30


def logistic(b, x, x0):
    if -b * (x - x0) > 100:
        logic = 0.9999999999999999
    else:
        logic = 1 / (1 + math.e ** (-b * (x - x0)))
    return logic


def invlogistic(b, y, x0):
    if y == 0:
        x = 0
    else:
        x = 1/b * math.log(abs(1 / y - 1)) + x0
    return x


def list_average(lst):
    if len(lst) == 0:
        return 0
    else:
        return sum(lst) / len(lst)


def lastNvalues(lst, N):
    res = list(islice(reversed(lst), 0, N))
    res.reverse()
    return res


def lastNaverage(lst, N):
    return list_average(lastNvalues(lst, N))


def anomalydetection(timeseries):
    predicted = timeseries[0]
    predictions = []
    deltasAveraged = []
    logDeltasNormalized = []
    deltas = []
    anomaly = []
    likelihood = 0
    standards = []
    temp = []

    for point in range(0, len(timeseries)):
        # calculating absolute value of deviation of prediction from actual point
        delta = abs(timeseries[point] - predicted)

        temp.append(timeseries[point])

        # append delta and prediction to their timeseries
        deltas.append(delta)
        predictions.append(predicted)

        lnA = lastNaverage(deltas, SLIDING_WINDOW)
        lnDn = lastNaverage(logDeltasNormalized, SLIDING_WINDOW)

        if lnA == 0:
            anomaly.append(0)
            deltasAveraged.append(0)
            logDeltasNormalized.append(0)
            standards.append(0)
        else:
            deltasAveraged.append(logistic(6, delta / lnA, 2.6))
            if lastNaverage(logDeltasNormalized, SLIDING_WINDOW) == 0:
                logDeltaNormalized = 0.01
            else:
                logDeltaNormalized = logistic(6 / lnDn, deltasAveraged[point], lnDn)

            likelihood = logistic(6, likelihood + logDeltaNormalized, 1)
            if likelihood > MILD_ANOMALY_THRESHOLD:
                temp.pop(-1)
                temp.pop(-1)
                temp.append(lastNaverage(temp, SLIDING_WINDOW))
                temp.append(lastNaverage(temp, SLIDING_WINDOW))

            logDeltasNormalized.append(logDeltaNormalized)
            anomaly.append(likelihood)

        predicted = 0.7 * (0.2 * temp[point] + 0.8 * timeseries[point]) + 0.3 * predicted


        # if point < SLIDING_WINDOW:
        #     standards.append(0)
        #     boundary.append(0)
        # else:
        #     standards.append(np.std(lastNvalues(timeseries[:point], 5)))
        #     boundary.append(np.mean(lastNvalues(standards[:point], SLIDING_WINDOW)))

    return anomaly, temp


def get_anomaly(timeseries, name):
    anomalydots = []
    mildanomalydots = []
    xdots = []
    xmilddots = []
    timerange = []

    for i in range(0, len(timeseries)):
        timerange.append(i)

    anomaly, temp = anomalydetection(timeseries)

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

    sp3.title.set_text('Anomaly likelihood')
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0.25)

    sp1.plot(timerange, timeseries, '-', color="silver", alpha=0.7, linewidth = 2.5)
    sp1.plot(timerange, xdots, '.', color="maroon", linewidth = 3)
    sp1.plot(timerange, xmilddots, '.', color="darkred", linewidth = 3)

    sp2.plot(timerange, temp, '.', color="silver", alpha=0.7, linewidth = 1.5)
    sp2.plot(timerange, xdots, '.', color="maroon", linewidth = 3)
    sp2.plot(timerange, xmilddots, '.', color="darkred", linewidth = 3)

    sp3.set_ylim(0, 1.0)
    sp3.plot(timerange, anomaly, '-', color="silver", alpha=0.7)
    # sp3.plot(timerange, anomalydots, 'r.')
    # sp3.plot(timerange, mildanomalydots, '.', color="orange")

    # sp2.fill_between(timerange, list(map(lambda x, y: x - y, predictions, boundary)), list(map(lambda x, y: x + y, predictions, boundary)), facecolor='lightblue')

    # plt.savefig(name + '.png', dpi=1200)
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

    get_anomaly(timeseries[0], 'Timeseries 0: Simple anomaly')

    get_anomaly(timeseries[1], 'Timeseries 1: ')

    get_anomaly(timeseries[2], 'Timeseries 2')
    # get_anomaly(timeseries[2][0:580], 'Timeseries 2: points 0..580')

    get_anomaly(timeseries[3], 'Timeseries 3')

    get_anomaly(timeseries[4], 'Timeseries 4')

    get_anomaly(timeseries[5], 'Timeseries 5')

    get_anomaly(timeseries[6], 'Timeseries 6')
    # get_anomaly(timeseries[6][:150], 'Timeseries 6: points 0..150')

    get_anomaly(timeseries[7], 'Timeseries 7')

    get_anomaly(timeseries[8], 'Timeseries 8')

    get_anomaly(timeseries[9], 'Timeseries 9: Numenta Anomaly Benchmark - Artificial no Anomaly no noise')

    get_anomaly(timeseries[10], 'Timeseries 10: Numenta Anomaly Benchmark - Artificial no Anomaly small noise')

    get_anomaly(timeseries[11], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise')
    # get_anomaly(timeseries[11][:500], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise Zoomed 0:500 points')
    # get_anomaly(timeseries[11][750:1000], 'Timeseries 11: Numenta Anomaly Benchmark - Artificial no Anomaly Noise Zoomed 750:1000 points')

    get_anomaly(timeseries[12], 'Artificial ANOMALY down')
    # get_anomaly(timeseries[12][2500:3300], 'Artificial ANOMALY down zoomed')

    get_anomaly(timeseries[13], 'Artificial ANOMALY flat middle')
    # get_anomaly(timeseries[13][2000:3000], 'Artificial ANOMALY flat middle')

    get_anomaly(timeseries[14], 'Artificial ANOMALY increase spike density')

    get_anomaly(timeseries[15], 'Timeseries 15: Real Traffic ANOMALY speed')
    # get_anomaly(timeseries[15][2000:2500], 'Timeseries 15: Real Traffic ANOMALY speed 2000..2500')

    # get_anomaly(timeseries[16][2450:2650], 'Timeseries 16: Real Traffic ANOMALY speed zoomed')
    # get_anomaly(timeseries[16][16000:], 'Timeseries 16: Real Traffic ANOMALY speed points 16000...')

    get_anomaly(timeseries[17], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[17][:200], 'Real Traffic ANOMALY Travel Time :200')
    # get_anomaly(timeseries[17][200:600], 'Real Traffic ANOMALY Travel Time 200:600')
    # get_anomaly(timeseries[17][600:1000], 'Real Traffic ANOMALY Travel Time 600:1000')

    get_anomaly(timeseries[18], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[18][:500], 'Real Traffic ANOMALY Travel Time')

    get_anomaly(timeseries[19], 'Real Traffic ANOMALY Travel Time')
    # get_anomaly(timeseries[19][1050:1300], 'Real Traffic ANOMALY Travel Time')

    get_anomaly(timeseries[20], 'Figure 3')
    # get_anomaly(timeseries[20][:300], 'Figure 2')
