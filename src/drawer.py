import os
import logging
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.algorithm import ALGO_TYPE

COLOR_MAP = {
    -1: 'indianred',
    0: 'orange',
    1: 'khaki',
    2: 'dodgerblue',
    3: 'lightgreen',
    4: 'slategray',
    5: 'paleturquoise',
    6: 'mediumpurple',
    7: 'lightsteelblue',
    8: 'hotpink',
    9: 'silver',
}


def init_plt(name, xlabel, ylabel):
    plt.clf()
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid()

def part1_drawer(name, train_data, test_data, algorithms):
    init_plt(name, 'x', 'y')

    trainx, trainy = train_data[0], train_data[1]
    testx, testy = test_data[0], test_data[1]
    
    # draw train data
    logging.info('drawing train data...')
    plt.scatter(trainx, trainy, label='train data', marker='.', s=7, c=COLOR_MAP[-1])

    # draw ground true curve
    logging.info('drawing ground true...')
    plt.plot(testx, testy, label='ground true', c=COLOR_MAP[-1], lw=1)

    # draw regression curve
    logging.info('drawing regression curves...')
    for idx, algorithm in enumerate(algorithms):
        mse_loss = algorithm.mse_loss(testx, testy)     # calculate prediction inside function
        objective_loss = algorithm.objective_loss(testx, testy)
        prediction = algorithm.predict(testx)

        if algorithm.algo_type == ALGO_TYPE.BR:
            yerr = np.array([prediction[1][val][val] for val in range(0, prediction[1].shape[0])])
            plt.errorbar(testx, prediction[0], yerr=yerr, fmt='-.', c=COLOR_MAP[idx], lw=1)
            plt.plot(testx, prediction[0], label=algorithm.name + ' mse=' + str(round(mse_loss, 3)), c=COLOR_MAP[idx], lw=1)
        else:     
            plt.plot(testx, prediction, label=algorithm.name + ' mse=' + str(round(mse_loss, 3)), c=COLOR_MAP[idx], lw=1)

    plt.legend(loc='best')
    file_path = 'output/part1/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path += name + '.jpg'
    plt.savefig(file_path, dpi=300)
    plt.close('all')


def part2_drawer(name, train_data, test_data,  mean, algorithms):
    init_plt(name, 'sample_id', 'people_num')

    trainx, trainy = train_data[0], train_data[1]
    testx, testy = test_data[0], test_data[1]
    sample_id = np.array([_ for _ in range(0, testy.shape[0])])

    # draw ground true
    plt.plot(sample_id, testy + mean, label='ground true', c=COLOR_MAP[-1], lw=1)

    for idx, algorithm in enumerate(algorithms):
        mse_loss = algorithm.mse_loss(testx, testy)
        prediction = algorithm.predict(testx)

        if algorithm.algo_type == ALGO_TYPE.BR:
            yerr = np.array([prediction[1][val][val] for val in range(0, prediction[1].shape[0])])
            plt.errorbar(sample_id, prediction[0] + mean, yerr=yerr, fmt='-.', c=COLOR_MAP[idx], lw=1)
            plt.plot(sample_id, prediction[0] + mean, label=algorithm.name + ' mse=' + str(mse_loss), c=COLOR_MAP[idx], lw=1)
        else:
            plt.plot(sample_id, prediction + mean, label=algorithm.name + ' mse=' + str(mse_loss), c=COLOR_MAP[idx], lw=1)
    
    plt.legend(loc='best')
    file_path = 'output/part2/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path += name + '.jpg'
    plt.savefig(file_path, dpi=300)
    plt.close('all')