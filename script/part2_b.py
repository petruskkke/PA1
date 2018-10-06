from src.drawer import *
from src.algorithm import *
from src.dataloader import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('PA1 - part2 - b')

    param_num = 18
    transformation = order_2nd_transformation

    logging.info('reading data...')
    data = mat_data_loader('dataset', part_id=2)
    trainx, trainy = data['trainx'], data['trainy']
    testx, testy = data['testx'], data['testy']
    meany = data['ym']
    
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
    lasso = L1RegularizedLS(transformation, param_num, name='lasso', alpha=100)
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
    part2_drawer(
        'part2_b_trans=' + str(int(param_num / 9)) + 'nd_order',
        [trainx, trainy],
        [testx, testy],
        meany,
        algorithms)
    
    logging.info('done.')
        