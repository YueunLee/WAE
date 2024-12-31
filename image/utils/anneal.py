import math

# No penalty annealing
def none_anneal(epoch:int, lamb: float, max_epochs: int):
    return lamb

'''
    Penalty annealing in linear rate;
    lamb = lamb * epoch
'''
def linear_anneal(epoch: int, lamb: float, max_epochs: int):
    alpha = 2.0 # 1.0
    return lamb + alpha * epoch

'''
    Penalty annealing in exponential rate;
    lamb = lamb * exp(c * (epoch-1))
    
    The constant c is set by a constant such that updated lamb at max_epoch equals to lamb * max_epochs.
'''
def exponential_anneal(epoch: int, lamb: float, max_epochs: int):
    c = math.log(max_epochs) / (max_epochs - 1) if max_epochs > 1 else 0.0
    return lamb * math.exp(c * (epoch - 1))