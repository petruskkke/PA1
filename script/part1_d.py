from src.drawer import *
from src.algorithm import *
from src.dataloader import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('PA1 - part1 - d')

    param_num = 6
    transformation = order_transformation
    outliersx = [11]
    logging.info('outlierxs: {0}'.format(outliersx))

    logging.info('reading data...')
    trainx, trainy = txt_data_loader('dataset', part_id=1, mode='train')
    testx, testy = txt_data_loader('dataset', part_id=1, mode='test')
    gt_param = txt_data_loader('dataset', part_id=1, mode='param')
    
    logging.info('add outliers...')
    ground_true = Algorithm(transformation, param_num, name='ground true')
    ground_true.param = gt_param
    superx, supery, outliersy = add_outliers(ground_true, trainx, trainy, 5, outliersx)
    outliers = list(zip(outliersx, outliersy))

    algorithms = []
    logging.info('doing LeastSquares Regression...')
    ls = LeastSquares(transformation, param_num, name='ls')
    ls.regress(superx, supery)
    algorithms.append(ls)

    logging.info('doing RegularizedLS Regression...')
    rls = RegularizedLS(transformation, param_num, name='rls', alpha=0.5)
    rls.regress(superx, supery)
    algorithms.append(rls)

    logging.info('doing L1RegularizedLS Regression...')
    lasso = L1RegularizedLS(transformation, param_num, name='lasso', alpha=0.01)
    lasso.regress(superx, supery)
    algorithms.append(lasso)

    logging.info('doing Robust Regression...')
    rr = RobustRegression(transformation, param_num, name='rr')
    rr.regress(superx, supery)
    algorithms.append(rr)

    logging.info('doing Bayesian Regression...')
    br = BayesianRegression(transformation, param_num, name='br', alpha=1, sigma=1)
    br.regress(superx, supery)
    algorithms.append(br)
    
    logging.info('drawing picture...')
    part1_drawer(
        'part1_d_outliders: {0}'.format(outliers),
        [trainx, trainy],
        [testx, testy],
        algorithms)
    
    logging.info('done.')
        