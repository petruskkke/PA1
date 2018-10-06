import numpy as np

from src.drawer import *
from src.algorithm import *
from src.dataloader import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('PA1 - part3 - a_1')

    param_num = 18
    transformation = order_2nd_transformation
    hypermethod = n_folder_cross_validation
    n_folder = 3
    
    logging.info('reading data...')
    data = mat_data_loader('dataset', part_id=2)
    trainx, trainy = data['trainx'], data['trainy']
    testx, testy = data['testx'], data['testy']
    meany = data['ym']
    
    n_dataset = create_n_fold_dataset(trainx, trainy, n=n_folder, shuffle=True)

    algorithms = []
    # self setting hyperparameters
    # ---------------------------------------------------
    logging.info('doing RegularizedLS Regression...')
    rls = RegularizedLS(transformation, param_num, name='rls', alpha=0.5)
    rls.regress(trainx, trainy)
    algorithms.append(rls)

    logging.info('doing L1RegularizedLS Regression...')
    lasso = L1RegularizedLS(transformation, param_num, name='lasso', alpha=100)
    lasso.regress(trainx, trainy)
    algorithms.append(lasso)

    logging.info('doing Bayesian Regression...')
    br = BayesianRegression(transformation, param_num, name='br', alpha=1, sigma=1)
    br.regress(trainx, trainy)
    algorithms.append(br)
    # ---------------------------------------------------


    # auto setting hyperparameters
    # ---------------------------------------------------
    logging.info('Find best hyperparams for RegularizedLS Regression...')
    search_rls = RegularizedLS(transformation, param_num, name='search-rls', alpha=0.5)
    rls_best = hypermethod(search_rls, n_dataset, np.arange(0, 10, 0.1))
    logging.info('Best hyperparams: {0}'.format(rls_best))
                                                             
    search_rls.hyperparams = rls_best
    search_rls.regress(trainx, trainy)
    algorithms.append(search_rls)

    logging.info('Find best hyperparams for L1RegularizedLS Regression...')
    search_lasso = L1RegularizedLS(transformation, param_num, name='search-lasso', alpha=100)
    lasso_best = hypermethod(search_lasso, n_dataset, np.hstack(np.arange(0, 100, 1)))
    logging.info('Best hyperparams: {0}'.format(lasso_best))    

    search_lasso.hyperparams = lasso_best
    search_lasso.regress(trainx, trainy)
    algorithms.append(search_lasso)

    logging.info('Find best hyperparams Bayesian Regression...')
    search_br = BayesianRegression(transformation, param_num, name='search-br', alpha=1, sigma=1)
    params = []
    for alpha in np.arange(1, 100, 1):
        for sigma in np.arange(1, 10, 1):
            params.append([alpha, sigma])
    br_best = hypermethod(search_br, n_dataset, params)
    logging.info('Best hyperparams: {0}'.format(br_best))

    search_br.hyperparams = br_best
    search_br.regress(trainx, trainy)
    algorithms.append(search_br)
    # ---------------------------------------------------
    
    logging.info('drawing picture...')
    part3_drawer(
        'part3_a_n_folder={0}'.format(n_folder),
        [trainx, trainy],
        [testx, testy],
        meany,
        algorithms)
    
    logging.info('done.')
    