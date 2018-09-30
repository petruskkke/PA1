import drawer
from dataloader import *
from algorithm import *

data = mat_data_loader('dataset', 2)
trainx, trainy = data['trainx'], data['trainy']
testx, testy = data['testx'], data['testy']
meany = data['ym']

param_num = 9
algorithms = []

ls = LeastSquares(identity_transformation, param_num, name='ls')
ls.regress(trainx, trainy)
algorithms.append(ls)

rls = RegularizedLS(identity_transformation, param_num, name='rls', alpha=0.5)
rls.regress(trainx, trainy)
algorithms.append(rls)

lasso = L1RegularizedLS(identity_transformation, param_num, name='lasso', alpha=100)
lasso.regress(trainx, trainy)
algorithms.append(lasso)

rr = RobustRegression(identity_transformation, param_num, name='rr')
rr.regress(trainx, trainy)
algorithms.append(rr)


br = BayesianRegression(identity_transformation, param_num, name='br', alpha=1, sigma=1)
br.regress(trainx, trainy)
algorithms.append(br)
    

drawer.part2_drawer(
        'part2',
        [trainx, trainy],
        [testx, testy],
        meany,
        algorithms)
