import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import beta, norm 


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

class LC_RW:

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
        self.sr     = self.args.params[3]
        self.eta    = self.args.params[4]
        self.lambba = self.args.params[5]

    def _init_RW_model( self):
        self.RW = np.zeros( [ self.dim,  self.args.K]) \
                  + self.args.w0     # DxK

    def _init_latent_casue( self):
        self.Z  = np.zeros( [self.T, self.args.K,])
        self.scale = np.ones( [ self.T, self.T]) ** self.g 

    def p_z1H( self): 
        # get p(Zt|Z1:t-1); H means history
        p_zH = self.scale[ :self.t, [self.t]] @ self.Z[ :self.t, :]  # 1xK : 
        p_zH[0, np.where(np.sum( self.Z[:self.t, :], axis=1
                        )==0)[0][0]] = self.alpha                     # 1xK
        p_z1H = p_zH / np.sum( p_zH)
        return p_z1H.T                                                # Kx1
        
    def p_xt1z( self, X):
        # get p(Xt|Z); X means stimuli
        # p(Xt|Z) = Prod_d p(xd|Z)
        N = np.sum( self.Z[:self.t, :], axis=1)
        xsum = X[:self.t, :].T @ self.Z[ :self.t, :]                  # DXK
        nu = self.sx / (N + self.sx) + self.sx                        # 1 
        p_x1z = 1.
        for d in range( xsum.shape[0]):
            xhat = xsum[ [d], :] / ( N + self.sx)
            p_x1z *= norm.pdf( X[self.t, d], xhat, np.sqrt(nu))
        return p_x1z                                                  # 1XK

    def p_r1xtz( self, X):
        # get p(R|Xt,Z)
        mu_r1xtz = X[ [self.t], :] @ self.RW         # 1xD x DXK = 1XK
        return mu_r1xtz, np.sqrt(self.sr)          # 1xK

    def p_z1xt( self, X):
        # get p(z|Xt)
        p_xtz = self.p_z1H().T * self.p_xt1z(X)    # 1xK ⊙ 1xK
        p_z1xt = p_xtz / np.sum(p_xtz)             # 1xK
        return p_z1xt.T                            # Kx1

    def p_z1rtxt( self, X, r):
        # p(z|xt,rt) ∝ p(z|xt)p(rt|xt,z)
        V, sr = self.p_r1xtz( X)
        p_rt1xtz = norm.pdf( r[self.t], V, sr)  # 1xK
        p_zrt1xt  = self.p_z1xt( X) * p_rt1xtz.T # Kx1 ⊙ Kx1
        p_z1rtxt  = p_zrt1xt / np.sum( p_zrt1xt) # Kx1
        return p_z1rtxt

    def update( self, X, r):
        
        while i < self.args.max_iter:  

            ## Inference
            p_z1xt = self.p_z1xt( X)
            V, _ = self.p_r1xtz( X)
            p_z1rtxt = self.p_z1rtxt( X, r)

            ## Optimize, this affect the future RW model 
            # update W = W + ηxtδ，δ = p(z|rt,xt) * (rt - V)
            rpe = p_z1rtxt * (r[ self.t] - V).T  # Kx1 ⊙ Kx1 = Kx1
            self.RW += self.eta * X[ [self.t], :] * rpe 

        # decide cluster, this affect the prior 
        k = np.argmax( p_z1rtxt, axis=0)
        self.Z[ self.t, k] = 1
        
        








            


    



        
        
        
         




        




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
        