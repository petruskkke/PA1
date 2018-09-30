from src.drawer import *
from src.algorithm import *
from src.dataloader import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('PA1 - part1 - e')

    param_num = 11
    transformation = order_transformation

    logging.info('reading data...')
    trainx, trainy = txt_data_loader('dataset', part_id=1, mode='train')
    testx, testy = txt_data_loader('dataset', part_id=1, mode='test')
    gt_param = txt_data_loader('dataset', part_id=1, mode='param')
    
    algorithms = []
    logging.info('doing LeastSquares Regression...')
    ls = LeastSquares(transformation, param_num, name='ls')
    ls.regress(trainx, trainy)
    algorithms.append(ls)

    logging.info('doing RegularizedLS Regression...')
    rls = RegularizedLS(transformation, param_num, name='rls', alpha=0.5)
    rls.regress(trainx, trainy)
    algorithms.append(rls)

    logging.info('doing L1RegularizedLS Regression...')
    lasso = L1RegularizedLS(transformation, param_num, name='lasso', alpha=0.01)
    lasso.regress(trainx, trainy)
    algorithms.append(lasso)

    logging.info('doing Robust Regression...')
    rr = RobustRegression(transformation, param_num, name='rr')
    rr.regress(trainx, trainy)
    algorithms.append(rr)

    logging.info('doing Bayesian Regression...')
    br = BayesianRegression(transformation, param_num, name='br', alpha=1, sigma=1)
    br.regress(trainx, trainy)
    algorithms.append(br)
    
    logging.info('drawing picture...')
    part1_drawer(
        'part1_e-order=' + str(param_num - 1),
        [trainx, trainy],
        [testx, testy],
        algorithms)
    
    logging.info('done.')
        