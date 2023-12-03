# https://github.com/jae0/HouseholderBPCA/blob/master/ubpca_improved.ipynb
# source: https://github.com/jae0/HouseholderBPCA/blob/master/py_stan_code/ppca_house_improved.stan

import numpy as np
import pystan
import pickle
from hashlib import md5
from sklearn import decomposition
from sklearn import datasets

Y = datasets.load_iris()["data"]

# For synthetic data
#####################################################
def haar_measure(D, Q):
    """
    outputs a matrix from stiefel(D, N)
    """
    z = np.random.normal(0,1,size=(D,D))
    q, r = np.linalg.qr(z)
    sign_r = np.sign(np.diag(r))
    return np.matmul(q, np.diag(sign_r))[:,:Q]
    
    
def get_data(N, D, Q, sigma):
    """
    U from stiefel(Q,D) and fixed sigma -> W = U*diag(sigma)
    X from normal 
    output Y = X*W.T, U
    """
    U = haar_measure(D,Q)
    W = np.matmul(U, np.diag(sigma))
    X = np.random.normal(size=(N, Q))
    return np.matmul(X, W.T) + np.random.normal(0, 0.01, size=(N, D)), U

#####################################################



# initialization with PCA solution
#####################################################
def sign_convention(U):
    """
    sign convention
    """
    return np.array( [-U[:,q] if U[0,q] < 0 else U[:,q] 
                         for q in range(U.shape[1])] ).T


def pca_solution(Y, Q):
    """returns first Q eigenvectors and eigenvalues of Y"""
    pca = decomposition.PCA(n_components=Q)
    pca.fit(Y)
    U_pca = pca.components_.T   
    U_pca = sign_convention(U_pca)
    sigma_pca = np.sqrt(pca.explained_variance_)
    return U_pca, sigma_pca


def householder(v: "array [Q,1]", D: int) -> "array [D,D]":
    """return householder transformation of size len(v) x len(v)"""
    Q = v.shape[0]
    H = np.eye(D,D)
    sgn = v[0,0]/np.fabs(v[0,0])
    u = v+sgn*np.linalg.norm(v)*np.eye(Q,1)
    u = u/np.linalg.norm(u)
    H[-Q:, -Q:] = -sgn*(np.eye(Q,Q) - 2*np.outer(u,u)) # -Q: means last Q items :: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
    return H


def get_v(U: "array [Q,Q]") -> "array [Q,1]":
    return U[:,0].reshape(-1,1)


def get_vs(U: "array [D,D]", D: int, Q: int) -> "list [list [q]]":
    """get the vs that leads to matrix U by inverse Householder trafos"""
    vs = []
    HU = U
    for q in range(Q):
        vs.append(get_v(HU[q:, q:]))
        H = householder(vs[-1], D)
        HU = H.dot(HU)
    return vs
 

def get_vs_stan_inp(U_pca: "array [D,Q]", D: int, Q: int) -> "list[floats]":
    """return the vs in a format that the stan code can take as input"""
    vs = get_vs(U_pca[:,::-1], D, Q) # stans ordering is reversed
    v_mat = np.zeros((D,Q))
    for i, v in enumerate(vs):
        v_mat[i:,i] = v.reshape(-1)
    vs_inp = []     # see function V_low_tri_plus_diag in stan code
    for d1 in range(D):
        for d2 in range(D):
            if d1 >= d2 and d2 < Q:
                vs_inp.append(v_mat[d1, d2])
    return vs_inp

#####################################################



def print_mat(A: "array [N,D]"):
    for row in A:
        for a in row:
            print("{:+.4f}".format(a), end="  ")
        print()
    print()


# from: http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html - modified
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'py_stan_code/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'py_stan_code/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm


```stan

functions{
    matrix V_low_tri_plus_diag_unnormed (int D, int Q, vector v) {
        // Put parameters into lower triangular matrix
        matrix[D, Q] V;

        int idx = 1;
        for (d in 1:D) {
            for (q in 1:Q) {
                if (d >= q) {
                    V[d, q] = v[idx];
                    idx += 1;
                } else
                V[d, q] = 0;
            }
        }
        return V;
    }
    matrix V_low_tri_plus_diag (int D, int Q, vector v) {
        matrix[D, Q] V = V_low_tri_plus_diag_unnormed(D, Q, v);
        for (q in 1:Q){
            V[,q] = V[,q]/sqrt( sum(square(V[,q])) );
        }
        return V;
    }
    real sign(real x){
        if (x < 0.0)
            return -1.0;
        else
            return 1.0;
    }
    matrix Householder (int k, matrix V) {
        // Householder transformation corresponding to kth column of V
        int D = rows(V);
        vector[D] v = V[, k];
        matrix[D,D] H;
        real sgn = sign(v[k]);
        
        //v[k] +=  sgn; //v[k]/fabs(v[k]);
        v[k] += v[k]/fabs(v[k]);
        H = diag_matrix(rep_vector(1, D)) - (2.0 / dot_self(v)) * (v * v');
        H[k:, k:] = -sgn*H[k:, k:];
        return H;
    }
    matrix[] H_prod_right (matrix V) {
        // Compute products of Householder transformations from the right, i.e. backwards
        int D = rows(V);
        int Q = cols(V);
        matrix[D, D] H_prod[Q + 1];
        H_prod[1] = diag_matrix(rep_vector(1, D));
        for (q in 1:Q)
            H_prod[q + 1] = Householder(Q - q + 1, V) * H_prod[q];
        return H_prod;    
    }
    matrix orthogonal_matrix (int D, int Q, vector v) {
        matrix[D, Q] V = V_low_tri_plus_diag(D, Q, v);
        // Apply Householder transformations from right
        matrix[D, D] H_prod[Q + 1] = H_prod_right(V);
        return H_prod[Q + 1][, 1:Q];    
    }
}
data{
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> Q;
    vector[D] Y[N];
}
transformed data{
    vector[D] mu = rep_vector(0, D);
}
parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    positive_ordered[Q] sigma;
    
    //vector[D] mu;
    real<lower=0> sigma_noise;
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    
    {
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;
        
        W = U*diag_matrix(sigma);
        
        K = W*W';
        for (d in 1:D)
            K[d, d] = K[d,d] + square(sigma_noise) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
    //mu ~ normal(0, 10);
    sigma_noise ~ normal(0,0.5);
    
    //v ~ normal(0,1);
    {
        matrix[D, Q] V = V_low_tri_plus_diag_unnormed(D, Q, v);
        for (q in 1:Q) {
            real r = sqrt(dot_self(V[,q]));
            r ~ gamma(100,100);
            target += -log(r)*(D-q);
        }
    }
    
    //prior on sigma
    target += -0.5*sum(square(sigma)) + (D-Q-1)*sum(log(sigma));
    for (i in 1:Q)
        for (j in (i+1):Q)
            target += log(square(sigma[Q-i+1]) - square(sigma[Q-j+1]));
    target += sum(log(2*sigma));
    
    Y ~ multi_normal_cholesky(mu, L);   
}
generated quantities {
    matrix[D, Q] U_n = orthogonal_matrix(D, Q, v);
    matrix[D, Q] W_n;
    
    //for (q in 1:Q)
        //if (U_n[1,q] < 0){
            //U_n[,q] = -U_n[,q];
        //}
    W_n = U_n*diag_matrix(sigma);
}
```

# Q=2
def fit(Y, Q, model, chains=4, iterations=2000):
    N, D = Y.shape
    Q=4

    #get PCA solution
    U_pca, sigma_pca = pca_solution(Y, Q)
    #W_pca = np.dot(U_pca,np.diag(sigma_pca))
    
    #get the vs by transforming U_pca to identity by Hs
    vs_stan_inp = get_vs_stan_inp(U_pca, D, Q)
    
    def init_fun():
        """Initialize with PCA solution"""
        return dict(
            v = vs_stan_inp,
            sigma = sigma_pca[::-1],   #ordering of stan is different (rev)
            mu = list(np.zeros(shape=D)),
            sigma_noise = 0.01
        )
    
    samples = run_stan(model, Y, N, D, Q, chains=chains, 
                       iterations=iterations, init=init_fun)
    return samples
