
"""
###################################
Als (``methods.factorization.als``)
###################################

"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *
import scipy.sparse as sp
import scipy.linalg as li 
import math

__all__ = ['Als']


class Als(smf.Smf):
    def __init__(self, V, seed=None, W=None, H=None, rank=30, max_iter=30,
                 lambda_var=0.1,
                 min_residuals=1e-5, test_conv=None, n_run=1, callback=None,
                 callback_init=None, track_factor=False, track_error=False,
                 Track=mf_track.Mf_track,
                 **options):
        self.name = "nmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        smf.Smf.__init__(self, vars())
        self.tracker = Track() if self.track_factor and self.n_run > 1 \
                                              or self.track_error else None
        self.lambda_var = lambda_var
        self.row_to_cols = {i: [] for i in range(V.get_shape()[0])}
        self.col_to_rows = {i: [] for i in range(V.get_shape()[1])}

        V_coo = V.tocoo()
        nonzero = V_coo.nonzero()
        for row, col in zip(nonzero[0], nonzero[1]):
            self.row_to_cols[row].append(col)
            self.col_to_rows[col].append(row)

        self.row_to_cols_num_2 = [
            math.sqrt(len(lst)) for _, lst in self.row_to_cols.items()]
        self.col_to_rows_num_2 = [
            math.sqrt(len(lst)) for _, lst in self.col_to_rows.items()]

        Vbin = V.copy()
        Vbin[Vbin != 0] = 1
        self.Vbin = Vbin

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            p_obj = c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(p_obj, c_obj, iter):
                p_obj = c_obj if not self.test_conv or iter % self.test_conv == 0 else p_obj
                self.update()
                self._adjustment()
                iter += 1
                if not self.test_conv or iter % self.test_conv == 0:
                    c_obj = self.objective()
                if self.track_error:
                    self.tracker.track_error(run, c_obj)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    run, W=self.W, H=self.H, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping
        parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param p_obj: Objective function value from previous iteration. 
        :type p_obj: `float`
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        pass
        # H = self.H
        # H[H <= 0.00001] = 0
        # self.H = H

        # W = self.W
        # W[W <= 0.00001] = 0
        # self.W = W

    def update(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        # self.H = multiply(
        #     self.H, elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div))
        # self.W = multiply(
        #     self.W, elop(dot(self.V, self.H.T), dot(self.W, dot(self.H, self.H.T)), div))
        rank = self.H.shape[0]

        self.W = self.W.tolil()
        for row, cols in self.row_to_cols.items():
            Hsub = self.H.tocsc()[:,cols]
            E = sp.identity(rank, dtype=np.float64)
            A = dot(Hsub, Hsub.T) + self.lambda_var * len(cols) * E
            Ainv = sp.csr_matrix(li.inv(A.toarray()), dtype=np.float64)
            Vsub = self.V.tocsr()[[row],:].tocsc()[:,cols]
            self.W[row,:] = dot(dot(Ainv, Hsub), Vsub.T).T
        self.W = self.W.tocsr()

        self.H = self.H.tolil()
        for col, rows in self.col_to_rows.items():
            Wsub = self.W.tocsr()[rows,:]
            E = sp.identity(rank, dtype=np.float64)
            A = dot(Wsub.T, Wsub) + self.lambda_var * len(rows) * E
            Ainv = sp.csr_matrix(li.inv(A.toarray()), dtype=np.float64)
            Vsub = self.V.tocsr()[rows,:].tocsc()[:,[col]]
            self.H[:,col] = dot(dot(Ainv, Wsub.T), Vsub)
        self.H = self.H.tocsr()

    def objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = multiply(self.V - dot(self.W, self.H), self.Vbin)
        W_2 = dot(sp.diags(self.row_to_cols_num_2, 0), self.W)
        H_2 = dot(self.H, sp.diags(self.col_to_rows_num_2, 0))

        R_sum = multiply(R, R).sum()
        W_sum = multiply(W_2, W_2).sum()
        H_sum = multiply(H_2, H_2).sum()
        # print(R_sum, W_sum, H_sum)
        return R_sum + self.lambda_var * ( W_sum + H_sum )

    def __str__(self):
        return '%s - update: %s obj: %s' % (self.name, self.update.__name__, self.objective.__name__)

    def __repr__(self):
        return self.name
