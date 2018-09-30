from src.drawer import *
from src.algorithm import *
from src.dataloader import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('PA1 - part1 - c')

    param_num = 6
    transformation = order_transformation
    runtime = 10
    subset = [0.25, 0.5, 0.75]
    logging.info('subset precent: {0}\t runtime for mean:{1}'.format(subset, runtime))

    logging.info('reading data...')
    trainx, trainy = txt_data_loader('dataset', part_id=1, mode='train')
    testx, testy = txt_data_loader('dataset', part_id=1, mode='test')
    gt_param = txt_data_loader('dataset', part_id=1, mode='param')
    
    for precent in subset:
        param_set = [
            np.zeros(param_num), np.zeros(param_num), np.zeros(param_num), np.zeros(param_num)]
        param_set.append([np.zeros(param_num), np.zeros((param_num, param_num))])
        
        algorithms = [
            LeastSquares(transformation, param_num, name='ls'),
            RegularizedLS(transformation, param_num, name='rls', alpha=0.5),
            L1RegularizedLS(transformation, param_num, name='lasso', alpha=0.01),
            RobustRegression(transformation, param_num, name='rr'),
            BayesianRegression(transformation, param_num, name='br', alpha=1, sigma=1),
        ]
        
        for idx in range(0, runtime):
            logging.info('run idx: {0}'.format(idx + 1))            

            logging.info('create subset: {0}%...'.format(precent))
            subx, suby = create_subset(trainx, trainy, precent=precent)

            logging.info('doing LeastSquares Regression...')
            algorithms[0].regress(subx, suby)
            param_set[0] += algorithms[0].param

            logging.info('doing RegularizedLS Regression...')
            algorithms[1].regress(subx, suby)
            param_set[1] += algorithms[1].param

            logging.info('doing L1RegularizedLS Regression...')
            algorithms[2].regress(subx, suby)
            param_set[2] += algorithms[2].param

            logging.info('doing Robust Regression...')
            algorithms[3].regress(subx, suby)
            param_set[3] += algorithms[3].param

            logging.info('doing Bayesian Regression...')
            algorithms[4].regress(subx, suby)
            param_set[4][0] += algorithms[4].param[0]
            param_set[4][1] += algorithms[4].param[1]

        for idx, algo in enumerate(algorithms):
            print(param_set[idx])
            if idx == 4:
                algo.param[0] = param_set[4][0] / runtime
                algo.param[1] = param_set[4][1] / runtime
            else:
                algorithms[idx].param = param_set[idx] / runtime
    
        logging.info('drawing picture...')
        part1_drawer(
            'part1_c_sub=' + str(precent) + '_run=' + str(runtime),
            [trainx, trainy],
            [testx, testy],
            algorithms)
        
    logging.info('done.')
        