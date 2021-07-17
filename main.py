import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import beta 


class Dirichlet:

    def __init__( self, dim, alpha=1, name=''):
        '''Init the dirichlet process

        dim: dimensions
        alph: init distribution parameters
        name: the semantic of each category 
        '''
        self.dim    = dim 
        self.params = np.ones( [dim,]) * alpha
        self.name   = name 

    def update( self, data):
        '''Update the distribution 

        data: the observation
        '''
        m = len( data)
        self.params[ :m] += data

    def rvs( self):
        '''sample a cat distribution
        '''
        p = np.random.gamma( self.params)
        return p / p.sum()

    def marginal_beta( self, i):
        '''Calculate the marginal beta of i config 
        '''
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return beta( alpha, alpha0 - alpha)

class LARW:

    def __init__( self, dim, T, args):
        '''Latent-cause rescolar-wagner model
        '''
        self.dim    = dim 
        self.T      = T
        self.args   = args
        self.t      = 0

    def _init_params( self):
        self.alpha  = self.args.params[0]
        self.g      = self.args.params[1]
        self.sx     = self.args.params[2]
        self.eta    = self.args.params[3]
        self.lambba = self.args.params[4]

    def _init_RW_model( self):
        self.RW = np.zeros( [ self.dim,  self.args.K]) \
                  + self.args.w0

    def _init_latent_casue( self):
        self.Z  = np.zeros( [self.T, self.args.K,])
        self.scale = np.ones( [ self.T, self.T]) ** self.g 

    def p_z1H( self): 
        # obtain p(Zt|Z1:t-1); H means history
        p_z1H = self.scale[ :self.t, [self.t]] @ self.Z[ :self.t, :]  # 1 x K
        p_z1H[0, np.where(np.sum( self.Z[:self.t, :], axis=1
                        )==0)[0][0]] = self.alpha                     # 1 x K
        p_z1H = p_z1H / np.sum( p_z1H)
        return p_z1H
        
    def p_x1z( self, X):
        # obtain p(Xt|Zt); X means stimuli
        # p(x|z) = Prod_d xd
        xsum = X[:self.t, :].T @ self.Z[ :self.t, :]                  # T X D
        nv = self.sx / (np.sum( self.Z[:self.t, :], axis=1
                         ) + self.sx) + self.sx                       # 1 
        p_x1z = 1.
        
        return xsum, nv 

    def p_r1z( self, p_x1z, p_z1H):
        # obtain p(rt|z) = ∑_x p(r|x)p(x|z)p(z) 
        # p(z)
        p_z = self.p_z1H()
        
        xsum, nv = self.p_x1z( X)
        for 
        
        
         




        




if __name__ == '__main__':

    dir = Dirichlet(3)
    names = [ 'lion', 'tiger', 'bear']
    for i in range(3):
        mar_dist = dir.marginal_beta(i)
        print( f'The percentage of {names[i]} in the prior is {mar_dist.mean():.02}.')

    data = [ 0.1, .2, .1]
    dir.update( data)

    for i in range(3):
        mar_dist = dir.marginal_beta(i)
        print( f'The percentage of {names[i]} in the prior is {mar_dist.mean():.02}.')
        