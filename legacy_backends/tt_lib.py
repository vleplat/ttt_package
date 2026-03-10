import tensorly as tl
import numpy as np
from numpy import einsum


from tensorly.tt_tensor import tt_to_tensor, TTTensor
from tensorly.tt_matrix import tt_matrix_to_tensor, TTMatrix

from tucker2_lib import (
    fast_tucker2_denoising,
    tucker2_truncatedhosvd_init,
    tucker2_to_tensor,
    minrank_boundedtrace,
    tucker_truncatedhosvd_init,
    tucker_denoising,
    tucker2_truncatedhosvd_init_complex_tensor,
    fast_tucker2_denoising_complex_tensor,
)



"""
tt_orth_at(x, pos, dir) : Orthogonalize single core
tt_orthogonalize(x, pos): Orthogonalize tensor
tt_nestedtk2(Y,approx_bound,precision = 1e-9):   TT as nested TK2
ttxt(Xtt, Y, mode, side):  TT-tensor Xtt time tensor Y along modes of Y to the left or right to
tt_adcu(X,Xt,rankX,opts):  Alternating two-cores update with left-right orthogonalization algorithm

ttmatrix_time_tt(Att,btt): TT-matrix times TT-tensor
ttmatrix_transpose(Att):   transpose a TT-matrix A
ttmatrix_AtA(Att):          A^T A :  A: TT-matrix
xQx_left(Qtt,xtt,mode):
xQx_right(Qtt,xtt,mode):
xQx_subset(Qtt,xtt,mode):   Subnet for Quadratic term x^T Q x
fx_left(ftt,xtt,mode):
fx_right(ftt,xtt,mode):
fx_subset(ftt,xtt,mode):  Subnet for linear term x^T f
      

Anh-Huy Phan
"""
# Yhat = tucker2_to_tensor(U1,U2,U3)
def tt_orth_at(x, pos, dir):
    """Orthogonalize single core.
    
    x = orth_at(x, pos, 'left') left-orthogonalizes the core at position pos
    and multiplies the corresponding R-factor with core pos+1. All other cores
    are untouched. The modified tensor is returned.
    
    x = orth_at(x, pos, 'right') right-orthogonalizes the core at position pos
    and multiplies the corresponding R-factor with core pos-1. All other cores
    are untouched. The modified tensor is returned.
    
    See also orthogonalize.
    """
    
    # Adapted from the TTeMPS Toolbox.
    
    Un = x[pos] # get the core at position pos
    if dir.lower() == 'left':
        
        Q, R = tl.qr(Un.reshape(-1, x.rank[pos+1]), mode='reduced') # perform QR decomposition
        # Fixed signs of x.U{pos},  if it is orthogonal.
        # This needs for the ASCU algorithm when it updates ranks both sides.
        sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
        Q = Q * sR # multiply Q by the signs
        R = (R.T * sR).T # multiply R by the signs
        
        # Note that orthogonalization might change the ranks of the core Xn
        # and X{n+1}. For such case, the number of entries is changed
        # accordingly. 
        # Need to change structure of the tt-tensor 
        # pos(n+1)
        #
        # update the core X{n}
        Un = Q.reshape(x.rank[pos], x.shape[pos], -1) # reshape Q to the core shape
        
        # update the core X{n+1}
        Un2 = R @ x[pos+1].reshape(x.rank[pos+1], -1) # multiply R by the next core
        
        # Check if rank-n is preserved 
        x[pos] = Un.reshape(x[pos].shape[0],x[pos].shape[1],R.shape[0]) # update the current core
        x[pos+1] = Un2.reshape(R.shape[0],x[pos+1].shape[1],-1) # update the next core
        
        if R.shape[0] != x.rank[pos+1]:
            x.rank = list(x.rank) # convert the tuple to a list
            x.rank[pos+1] = R.shape[0] # assign the new value
            x.rank = tuple(x.rank) # convert the list back to a tuple

    
    elif dir.lower() == 'right':
        # mind the transpose as we want to orthonormalize rows
        Q, R = tl.qr(Un.reshape(x.rank[pos], -1).T, mode='reduced') # perform QR decomposition on the transpose
        # Fixed signs of x.U{pos},  if it is orthogonal.
        # This needs for the ASCU algorithm when it updates ranks both
        # sides.
        sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
        Q = Q * sR # multiply Q by the signs
        R = (R.T * sR).T # multiply R by the signs
        
        Un = Q.T.reshape(-1, x.shape[pos], x.rank[pos+1]) # reshape Q transpose to the core shape
        Un2 = x[pos-1].reshape(-1, x.rank[pos]) @  R.T # multiply the previous core by R transpose
        
        x[pos] = Un.reshape(Un.shape[0],x[pos].shape[1],-1) # update the current core
        x[pos-1] = Un2.reshape(x[pos-1].shape[0],x[pos-1].shape[1],-1) # update the previous core
        
        if R.shape[0] != x.rank[pos]:
            x.rank = list(x.rank) # convert the tuple to a list
            x.rank[pos] = R.shape[0] # assign the new value
            x.rank = tuple(x.rank) # convert the list back to a tuple
    else:
        raise ValueError('Unknown direction specified. Choose either LEFT or RIGHT')
    
    return x

def tt_orthogonalize(x, pos):
    """Orthogonalize tensor.
    
    x = orthogonalize(x, pos) orthogonalizes all cores of the TTeMPS tensor x
    except the core at position pos. Cores 1...pos-1 are left-, cores pos+1...end
    are right-orthogonalized. Therefore,
    
    x = orthogonalize(x, 1) right-orthogonalizes the full tensor,
    
    x = orthogonalize(x, x.order) left-orthogonalizes the full tensor.
    
    See also orth_at.
    """
    
    # adapted from the TTeMPS Toolbox.
    
    # left orthogonalization till pos (from left)
    for i in range(pos):
        # print(f'Left orthogonalization {i}')
        x = tt_orth_at(x, i, 'left')
    
    # right orthogonalization till pos (from right)
    ndimX = len(x.factors)

    for i in range(ndimX-1, pos, -1):
        # print(f'Right orthogonalization {i}')
        x = tt_orth_at(x, i, 'right')
    
    return x



def parseInput(opts):
    """Set algorithm parameters from input or by using defaults 
    For TT algorithms
    Parameters
    ----------
    opts : dict
    A dictionary of options for the algorithm
    
    Returns
    -------
    param : dict
    A dictionary of parsed parameters
    
    """
    
    # Set algorithm parameters from input or by using defaults
    param = {}
    param["init"] = opts.get("init", "nvec")
    assert isinstance(param["init"], (list)) or param["init"][:4] in ["rand", "nvec", "fibe", "orth", "dtld", "exac"]
    param["maxiters"] = opts.get("maxiters", 200)
    param["tol"] = opts.get("tol", 1e-6)
    param["compression"] = opts.get("compression", True)
    param["compression_accuracy"] = opts.get("compression_accuracy", 1e-6)
    param["noise_level"] = opts.get("noise_level", 1e-6) # variance of noise
    param["exacterrorbound"] = opts.get("exacterrorbound", True) # |Y-Yx|_F = noise level
    param["printitn"] = opts.get("printitn", 0)
    param["core_step"] = opts.get("core_step", 2)
    assert param["core_step"] in [1, 2] # or 1: non-overlapping or overlapping update
    param["normX"] = opts.get("normX", None)
    return param

def tt_getrank(X):
    # X : array of core tensors of shape (r_k, n_k, r_{k+1})
    if len(X) == 0:
        raise ValueError("X must contain at least one core")
    tt_rank = [X[0].shape[0]] + [x.shape[2] for x in X]
    return np.array(tt_rank, dtype=int)



def lowrank_matrix_approx(T, error_bound, exacterrorbound=True):
    # find the best low-rank approximation to T
    # such that |T-Tx|_F^2 <= error_bound
    # Tx = u*diag(s)*v.T
    # 
    # Anh-Huy Phan
    # TENSORBOX, 2018
    
    # Perform singular value decomposition of T
    u, s, v = np.linalg.svd(T, full_matrices=False)
    # Convert s to a diagonal matrix
    #s = np.diag(s)
    # Compute the cumulative sum of squared singular values
    cs = np.cumsum(s**2)
    # Find the smallest rank r1 such that the approximation error is within the bound
    r1 = np.where((cs[-1] - cs) <= error_bound)[0][0]
    
    # Truncate u, s, and v to rank r1

    u = u[:, :r1+1]
    s = s[:r1+1]
    v = v[:r1+1, :].T
    # Compute the approximation error
    # approx_error = cs[-1] - cs[r1]
    approx_error = cs[-1] - cs[r1]

    # Save the original singular values
    s0 = s.copy()
    # print(f'Rank {r1}')
    
    if exacterrorbound:
        # Modify s to attain the predefined approximation error
        s[0] = s[0] + np.sqrt(error_bound - approx_error) # or s[0, 0] - np.sqrt(error_bound - approx_error)
        approx_error = error_bound
    
    return u, s, v, approx_error, s0



def tt_nestedtk2(Y,approx_bound,precision = 1e-9):
    # Nested TUCKER-2 for TT decomposition 
    # Y is approximated by a TT -tensor 
    # Ytt = Uleft_1 *  G1 * Uright_1 (Tucker-2 model)
    # G{k-1} = Uleft_k *  Gk * Uright_k 
    #
    # 
    # Anh-Huy Phan
    # 
    # approx_bound = noiselevel**2 * Y.size
    # precision = 1e-9 # small number added to the approximation error bound 
    normY2 = tl.norm(Y)**2
    szY = Y.shape
    N = Y.ndim
    rankTTx = np.zeros(N+1, dtype = 'int')
    rankTTx[0] = 1
    rankTTx[N] = 1

    
    Yk = Y   
    
    # Factors of the nested TT 
    Factors = [None] * N
    maxiters = 1000
    
    for k in range(np.array((N-N%2)/2,int)):
        # Layer-K 
        print(f'Perform the {k+1}-th Tucker-2 decomposition')
        # approximation error = \|Y - Ul(1:K)*Yxk*Ur(K:1)\|_F^2
        #  = \|Y\|_F^2 + \|Yxk\|_F^2 - 2<Yk, Yxk>
        #  = \|Yk - Yxk\|_F^2 - ||Yxk||_F^2 +\|Y\|_F^2 <= delta^2
        # 
        # \|Yk - Yxk\|_F^2 <= delta^2 - \|Y\|_F^2 +||Yk||_F^2
         
        if ((N%2)==0) and (k == (np.array((N-N%2)/2,int)-1)):
            # For the last level when N is an even number
            Yk = Yk.reshape(rankTTx[k]*szY[k],rankTTx[N-k]*szY[N-1-k])
            normYk = tl.norm(Yk)**2
            approx_bound_n = approx_bound - normY2 + normYk
            approx_bound_n = approx_bound_n + precision
         
            Ulk, s, Urk, approx_error,_ = lowrank_matrix_approx(Yk, approx_bound_n, exacterrorbound=False)
            Ulk = Ulk * s       
        
        else:
            # Tucker-2 decomposition 
            Yk = Yk.reshape(rankTTx[k]*szY[k],-1,rankTTx[N-k]*szY[N-1-k])
            normYk = tl.norm(Yk)**2
            approx_bound_n = approx_bound - normY2 + normYk
            approx_bound_n = approx_bound_n+precision
            noiselevel_n = np.sqrt(approx_bound_n/Yk.size)
            
            # initialization
            Ulk, Urk, err,_,_ = tucker2_truncatedhosvd_init(Yk,approx_bound_n)
            # Tucker2 decomposition with given approximation error 
            Ulk,Yknew,Urk,approx_error,noparams,rankR = fast_tucker2_denoising(Yk, [Ulk, Urk], maxiters, 1e-7, noiselevel_n, exacterrorbound=False, precision=1e-7, verbose=True) 
            
        
        # print(f'Approximation Error {approx_error[-1]} | Predefined Bound {approx_bound}')

        approx_error = normY2 - normYk + approx_error
        # Update ranks
        rankTTx[1+k] = Ulk.shape[1]
        rankTTx[N-1-k] = Urk.shape[1]
    
        # Update the left factor
        Factors[k] = Ulk.reshape(rankTTx[k],szY[k],rankTTx[k+1])
        # Update the right factor
        Factors[N-1-k] = Urk.T.reshape(rankTTx[N-1-k],szY[N-1-k],rankTTx[N-k]) 
        Yk  = Yknew
    
    if (N%2)==1:
        # core tensor in 
        k =np.array((N-N%2)/2,int)
        Factors[k] = Yk.reshape(rankTTx[k],szY[k],rankTTx[k+1])
    
    # for n in range(N):     
    #     Factors[n] = Factors[n].reshape(rankTTx[n],szY[n],rankTTx[n+1])
    
    Yttnested = TTTensor(Factors)
    # print(Yttnested)
    # approx_error = tl.norm(Y - tt_to_tensor(Yttnested))**2
    
    print(f'Approximation Error {approx_error} | Predefined Bound {approx_bound}')
    return Yttnested, approx_error



def ttxt(Xtt, Y, mode, side):
    # TT-tensor Xtt time tensor Y along modes of Y to the left or right to
    # "mode"
    #
    # Phan Anh Huy
    
    N = tl.ndim(Y)
    
    SzY = Y.shape
    # SzX = Xtt.shape
    rankX = Xtt.rank
    
    if side == 'left':
        modes = range(min(mode))
    
        if modes:
            for n in modes:
                if n == 0:
                    # if isinstance(Y, sptensor): # update on Feb 2nd
                    #     # Z = sptenmat(Y, 0)
                    #     # Z = sparse(Z)
                    #     Z = tl.reshape(Y, (SzY[0], np.prod(SzY[1:])))
                    #     Z = ttm(Z, tl.reshape(Xtt[n], (SzY[0], -1)).T, 0) # R2 x (I2 ... IN)
                    # else:
                    Z = tl.reshape(Y, (SzY[0], np.prod(SzY[1:])))
                    Z = tl.dot(tl.reshape(Xtt[n], (SzY[0], -1)).conj().T, Z) # R2 x (I2 ... IN)
                else:
                    # if isinstance(Y, sptensor): # update on Feb 2nd
                    #     # Z = sptenmat(Y, 0)
                    #     # Z = sparse(Z)
                    #     Z = tl.reshape(Z, (rankX[n] * SzY[n], np.prod(SzY[n+1:])))
                    #     Z = ttm(Z, tl.reshape(Xtt[n], (rankX[n] * SzY[n], -1)).T, 0) # R2 x (I2 ... IN)
                    # else:
                    Z = tl.reshape(Z, (rankX[n] * SzY[n], -1)) # In x (In+1... IN) R2 R3 ...R(n-1)
                    Z = tl.dot(tl.reshape(Xtt[n], (rankX[n] * SzY[n], -1)).conj().T, Z) # R2 x (I2 ... IN)
                Z = tl.reshape(Z, (rankX[n+1], *SzY[n+1:]))
        else:
            Z = Y
    
    elif side == 'right': # contract (N-n+1) cores
        modes = range(N-1, max(mode), -1)
    
        if modes:
            modes_X = range(N-1, N-len(modes)-1, -1)
            for n in range(len(modes)):
                nX = modes_X[n]
                nY = modes[n]
                
                if nY == N-1:
                    Z = tl.reshape(Y, (-1, SzY[N-1]))
                    
                    if tl.ndim(Xtt[nX]) == 2:
                        Z = tl.dot(Z, Xtt[nX].conj().T) # R2 x (I2 ... IN)
                    else: # Xn : rn x In x R
                        # When Xtt is a block TT, the last core is not a
                        # matrix but an order-3 tensor
                        
                        Z = tl.dot(Z, tl.reshape(tl.transpose(Xtt[nX].conj(), (1, 0, 2)), (Xtt[nX].shape[1], -1))) # R2 x (I2 ... IN)
                        Z = tl.transpose(tl.reshape(Z, (-1, Xtt[nX].shape[2])), (1, 0)) # R x I1 x ... In-1 x Rn
                else:
                    Z = tl.reshape(Z, (-1, rankX[nX+1] * SzY[nY])) # In x (In+1... IN) R2 R3 ...R(n-1)
                    Z = tl.dot(Z, tl.reshape(Xtt[nX].conj(), (-1, rankX[nX+1] * SzY[nY])).T) # R2 x (I2 ... IN)
            
            Z = tl.reshape(Z, (Xtt[N-1].shape[2], *SzY[:nY], rankX[nX]))
            Z = tl.transpose(Z, list(range(1, tl.ndim(Z))) + [0])
        else:
            Z = Y
    
    elif side == 'both':
        Z = ttxt(Xtt, Y, mode, 'right')
        Z = ttxt(Xtt, Z, mode, 'left')
    
    return Z




def tt_adcu(X,Xt,rankX,opts):
    # Define a function to perform alternating two-cores update with left-right orthogonalization algorithm
    # which approximates a tensor X by a TT-tensor of rank rank-X
    
    # def ttmps_a2cu(X, rankX, opts):
    # Each time of iteration, the algorithm updates two cores simultaneously,
    # then updates the next two cores. The update process runs from left to
    # right, i.e., from the first core to last core. Then it runs from right to
    # left to update cores in the descending order, i.e, N, N-1, ..., 2, 1
    
    # In general, the process is as the following order when N is odd
    
    #  Left to right : (1,2), (3,4), ..., (N-2,N-1),
    #  Right to left : (N,N-1), (N-2, N-3), ..., (3,2)
    #  Left to right : (1,2), (3,4), ..., (N-2,N-1)
    
    # When N is even, update orders are as follows
    
    #  Left to right : (1,2), (3,4), ..., (N-3,N-2),
    #  Right to left : (N,N-1), (N-2, N-3), ..., (4,3)
    #  Left to right : (1,2), (3,4), ..., (N-3,N-2),
    
    # Note, when approximating a tensor X by a TT-tensor with rank specified,
    # the algorithm often does not achieve solution after a few rounds of LR-LR
    # updates, and may need many iterations. This will require high
    # computational cost due to projection of the tensor on to subspace of
    # factor matrices of the estimation tensor.
    
    # To this end, instead of fitting the data directly, one should compress
    # (fit) X by a TT-tensor with higher accuracy, i.e. higher rank, then
    # truncate the TT-tensor to the rank-X using this algorithm.
    # This compression procedure is lossless when rank of the approximation
    # TT-tensor higher than the specified rank_X.
    
    # The algorithm supports compression option as an acceleration method
    
    #  Parameters
    
    #    compression: 1     1:  prior-compression of X to a TT-tensor. The
    #                       algorithm will manipulate on the TT-tensors
    
    #    compression_accuracy: 1.0000e-06  :  \|Y - X\|_F <= compression_accuracy
    
    #    init: 'nvec'  initialization method
    #    maxiters: 200     maximal number of iterations
    
    #    noise_level: 1.0000e-06   : standard deviation of noise,
    #                     if it is given, ASCU solves the denoising problem,
    #                     i.e.,   min  \|Y - X\|_F  <= noise_level^2 * numel(Y)
    
    #     normX: []    norm of the tensor X
    #
    # Anh-Huy Phan
    #    printitn: 0
    #    tol: 1.0000e-06

    def factorize_left_right_proj(modes, prev_n):
        # Truncated SVD of Tn
        # The full objective function is
        #  min  \|Y\|_F^2 - \|Tn\|_F^2 + \| Tn - G[n] * G[n+1] \|_F^2
        #
        # which achieves minimum when G[n] * G[n+1] is best rank-(Rn+1)
        # approximation to Tn.
        #       f_min =  \|Y\|_F^2 - sum(s**2)
        #  where s comprises R_n leading singular values of Tn.
        
        modes = sorted(modes)
        sv = []
        
        # For a tensor X, the contraction is computed through
        # a progressive computation, in which the left-contraced
        # tensors are saved
        if progressive_mode:
            # Update the Phi-Left if needed
            Phi_left = contract_update(Xt, X, Phi_left, prev_n, modes[0])
            # Compute the both-side contracted tensor
            Tn, Phi_left = contraction_bothsides(modes, Phi_left)
        else:
            Tn = ttxt(Xt, X, modes, 'both') # contraction except mode-n
            # print(f'Tn {Tn.shape}')
        
        Tn = tl.reshape(Tn, (rankX[modes[0]] * SzX[modes[0]], -1))
        
        # Factorize Tn
        # print(f'Noise level {param["noise_level"]}')
        if param["noise_level"] is None:
            
            u, s, v = np.linalg.svd(Tn, full_matrices=False)
            u = u[:, :rankX[modes[0] + 1]]
            v = v[:rankX[modes[0] + 1],:].T
            s = s[:rankX[modes[0] + 1]]
            # s = np.diag(s)
            #s = s[:rankX[modes[0] + 1], :rankX[modes[0] + 1]]
            sv = s
        
        else:
            normTn2 = np.linalg.norm(Tn)**2
        
            # When the noise level is given, A2CU solves the denoising
            # problem
            # min \| Y - X\|_F^2 = |Y|_F^2 - |Tn|_F^2 + |T_n-X|_F^2 <= eps
            #
            # i.e.
            #   min   |T_n-X|_F^2  <=  eps_n
            #
            #  where the accuracy level eps_n is given by
            
            accuracy_n = tt_accuracy + normTn2 # eps - |Y|_F^2 + |T_n|_F^2
            # print(f'accuracy_n {accuracy_n}, solve denoising problem')
            if accuracy_n < 0:
                # Negative accuracy_n indicates that the rank is too small
                # to explain the data with the given accuracy error.
                # Rank of the core tensor needs to increase
                
                # Tn = tl.reshape(Tn, (rankX[modes[0]] * SzX[modes[0]], -1))
                u, s, v = np.linalg.svd(Tn, full_matrices=False)
                u = u[:, :rankX[modes[0] + 1]]
                v = v[:rankX[modes[0] + 1],:].T
                # s = np.diag(s)
                #s = s[:rankX[modes[0] + 1], :rankX[modes[0] + 1]]
                s = s[:rankX[modes[0] + 1]]
                sv = s
            
            else: # accuracy_n >  0 # Solve the denoising problem
                # Tn = tl.reshape(Tn, (rankX[modes[0]] * SzX[modes[0]], -1))
                # u and v are singular vectors of the Tn matrix,
                # Tn_appx = u*diag(s) * v.T ,# sv are singular values
                # print(f'Tn {Tn.shape}')
                u, s, v, approx_error_Tn, sv = lowrank_matrix_approx(Tn, accuracy_n, param["exacterrorbound"])

                # print(f'u {u.shape}, v {v.shape}, s {s.shape}')
        return u, s, v, Tn, sv
    
    # Fill in optional variable
    if "opts" not in locals():
        opts = {}
    
    param = parseInput(opts)
     
    # if len(sys.argv) == 0:
    #     Xt = param
    # return Xt
    
    # Print message
    if param["printitn"] != 0:
        print("\nAlternating Double-Cores Update for TT-decomposition\n")
    
    # Correct ranks of X
    N = X.ndim
    SzX = X.shape
        
    if rankX is not None:
        for n in list(range(1, N)) + list(range(N-1, 0, -1)):
            rankX[n] = min([rankX[n], rankX[n-1]*SzX[n-1], rankX[n+1]*SzX[n]])
    
    # Stage 1: compress the data by a TT-tensor using SVD truncation
    maxiters = param["maxiters"]
    
    # Compress data to TT-tensor with specific accuracy, not by specified rank
    # the compression_accuracy should be at least equal to the noise level
    
    # if param["compression|]:
    #     if not isinstance(X, tt_tensor):
    #     if param["compression_accuracy"] is None:
    #     param["compression_accuracy"] = param["noise_level"]
    #     X = tt_tensor(X, param.compression_accuracy, SzX, rankX)
    #      # adjust the norm X in the cost function due to compression
    #     # normX2 = normX2 + 2*norm(Xtt)**2 - 2 * innerprod(Xtt,X)
    #     # X = Xtt
    
    # Precompute norm of the tensor X, which is used for fast assessment of the
    # cost function
    if param["normX"] is not None:
        normX2 = param["normX"]**2
    else:
        normX2 = tl.norm(X)**2
     
    # #  Get initial value or Initialize a TT-tensor by rouding X
    # Xt = initialization;
    
    #  Output is a tensor orthogonalized from right to left
    Xt = tt_orthogonalize(Xt,0)
    rankX = Xt.rank # rank of Xt may change due to the orthogonalization
    rankX = np.array(rankX)
    
    err = np.array(np.nan) # array of nan values with shape (maxiters,)
    sum_rankX = np.array(np.nan) # array of nan values with shape (maxiters,)
    prev_error = None
    # stop_  = False
    tol = param["tol"]
    cnt = 0
    
    # Core indices in the left-to-right and right-to-left update procedures
    # core_step = 1 or 2
    left_to_right_updims = list(range(0, N - param["core_step"], param["core_step"])) # left to right update mode
    if N % 2 == 0:
        right_to_left_updims = [N-1] + list(range(N-2, param["core_step"]-1, -param["core_step"])) # right to left update mode
    else:
        right_to_left_updims = list(range(N-1, 0, -param["core_step"])) # right to left update mode
    
    no_updates_L2R = len(left_to_right_updims)
    no_updates_R2L = len(right_to_left_updims)
    
    # Expand the core index arrays by the first index of the other one
    left_to_right_updims = left_to_right_updims + [right_to_left_updims[0]]
    right_to_left_updims = right_to_left_updims + [left_to_right_updims[0]]
    
    # Precompute left- and right contracted tensors
    # progressive_mode = True # for progressive computation of contracted tensors Tn
    progressive_mode = False
    if progressive_mode:
        Phi_left = [None] * N # cell(N,1)
        Phi_left[0] = 1
    
    # Main part
    max_stop_cnt = 6
    stop_cnt = 0
    
    if param["noise_level"] is not None:
        # If noise_level is given the algorithm solves the denoising problem
        #     \| Y - X \|_F^2 < noise_level
        # such that X has minimum rank
        tt_rank_determination = True # track the estimated ranks
        tt_accuracy = param["noise_level"]**2 * np.prod(SzX)
        tt_accuracy = tt_accuracy - normX2
    
    prev_n = 0 # previous core index
 
    
    for kiter in range(maxiters):
    
        # Run the left to right update
        # This round will update pair of cores (1,2), (3,4), ...,
        # or (1,2),(2,3),... which depends on the core_step
        
        for k_update in range(no_updates_L2R):
            # Core to be updated
            n = left_to_right_updims[k_update] # core to be updated
            
            # next_n: the next core to be updated
            next_n = left_to_right_updims[k_update + 1]
    
            # print(f'Cores to be updated {n, n+1}')
            # counter
            cnt += 1
            
            # Update the core X[n] and X[n+1]
            
            # Truncated SVD of projected data Tn
            # The full objective function is
            #  min  \|Y\|_F^2 - \|Tn\|_F^2 + \| Tn - X[n] * X[n+1] \|_F^2
            #
            # where Tn is left right projection of X by cores of Xt except two
            # cores  n and n+1.
            #
            # which achieves minimum when X[n] * X[n+1] is best rank-(Rn+1)
            # approximation to Tn
            # The objective function is
            #       f_min =  \|Y\|_F^2 - sum(s.^2)
            #
            # where s comprises R_n leading singular values of Tn.
            # modes: arrays of core indices
            modes = [n, n+1]
            u, s, v, Tn, sv = factorize_left_right_proj(modes, prev_n)
            
            # Assess the approximation error - objective function
            # curr_err = (normX2 - norm(Tn,'fro')**2 +  norm(Tn - u*diag(s)*v.T,'fro')**2)/normX2
            # curr_err = tl.norm(tt_to_tensor(Xt) - X)**2/normX2
            if param["exacterrorbound"] == 1:
                curr_err = (normX2 + s.T @ (s - 2 * sv)) / normX2 # relative error
            else:
                curr_err = (normX2 - np.sum(s**2)) / normX2
            # print(f'curr_err {curr_err}')
            # curr_err = (normX2 - np.sum(s**2)) / normX2
            # print(f'Modes {modes} {[curr_err, curr_err2]}')
            # err = np.append(err,curr_err)
            
            # Update core tensors X[n] and X[n+1]
            # Update Xn

            Xt.factors[n] = tl.reshape(u, (rankX[n], SzX[n], -1))
            
            # Since u is orthogonal, do not need to left-orthogonalize X[n]
            # Update rank of X[n]
            rnp1 = u.shape[1] # new rank R(n+1) 
            rankX[n+1] = rnp1
            #Xt.rank[n+1] = rnp1 # Xt.rank is a tuple, cannot change the value of a tuple element
            Xt.rank = rankX
 
            sum_rankX = np.append(sum_rankX,np.sum(rankX))
            
            # Check convergence
            if (prev_error is not None) and (np.abs(curr_err - prev_error) < tol) and (sum_rankX[cnt] == sum_rankX[cnt-1]):
                stop_cnt += 1
                # break
            else:
                stop_cnt = 0
                
            
            
             
            # If the next core to be updated is not X[n+1], but e.g., X[n+2],
            # Update X[n+1], and left-orthogonalize this core tensor
 
            
            v = v @ np.diag(s) # or v * s
             
            if next_n >= (n+2): # then next_n == n+2
                # left-Orthogonalization for U[n+1]
                v = tl.reshape(v.T, (rnp1 * SzX[n+1], -1))
                v, vR = tl.qr(v, mode="reduced")
                
                rnp2 = v.shape[1]
 
                 
                # no need adjust Xt[n+2] because it will be updated in next iteration
                # except the last iteration
                # if k_update == no_updates_L2R:
                Xt.factors[n+2] = tl.reshape(vR @ tl.reshape(Xt.factors[n+2], (rankX[n+2], -1)), (rnp2, SzX[n+2], -1))
                
                # Update X[n+1]
                Xt.factors[n+1] = tl.reshape(v, (rnp1, SzX[n+1], -1))
                
                # Update rank of X[n+1]
                rankX[n+2] = rnp2
                #Xt.rank[n+2] = rnp2
                Xt.rank = rankX
            
            else: # if n_next = n+1
                # Update X[n+1]
                # if k_update == no_updates_L2R:
                Xt.factors[n+1] = tl.reshape(v.T, (rnp1, SzX[n+1], -1))


            # # for debugging
            # curr_err2 = tl.norm(tt_to_tensor(Xt) - X)**2/normX2
            # print(f'Modes {modes} {[curr_err, curr_err2]}')
            err = np.append(err,curr_err)

            prev_error = curr_err
            if (param["printitn"]>0) and (cnt % param["printitn"] == 0):            
                print(f"Iter {kiter}, Cores {modes}, Error {curr_err:.5e}, SumRank {sum_rankX[cnt]}")

 
            prev_n = n
            
            if stop_cnt > max_stop_cnt:
                # If the left-to-right update stops, then
                # update X[n+2]
                # if next_n >= (n+2):
                #     Xt.factors[n+2] = tl.reshape(vR @ tl.reshape(Xt.factors[n+2], (rankX[n+2], -1)), (rnp2, SzX[n+2], -1))
                # else:
                #     Xt.factors[n+1] = tl.reshape(v, (rnp1, SzX[n+1], -1))
                break
        # end of for L2R loop  
        
        if stop_cnt > max_stop_cnt:
            break 
        
       
        # right to left update
        # Last Phi_left which is updated is Phi_left(modes(2))
        for k_update in range(no_updates_R2L):
 
            n = right_to_left_updims[k_update] # core to be updated
            next_n = right_to_left_updims[k_update + 1] # next core to be updated
            
            cnt += 1
            
            # Factorization of left and right projection matrix
            #  of size (Rn In)  x (I(n+1)  R(n+2))
            modes = [n, n-1]

            # print(f'Cores to be updated {n, n-1}')
            
            # u, s, v = factorize_left_right_proj(modes, prev_n)
            u, s, v, Tn, sv = factorize_left_right_proj(modes, prev_n)
            
            # Assess the approximation error - objective function
            # curr_err = (normX2 - norm(Tn,'fro')**2 +  norm(Tn - u*diag(s)*v.T,'fro')**2)/normX2
            # curr_err = tl.norm(tt_to_tensor(Xt) - X)**2/normX2
            if param["exacterrorbound"] == 1:
                curr_err = (normX2 + s.T @ (s - 2 * sv)) / normX2
            else:
                curr_err = (normX2 - np.sum(s**2)) / normX2
            
            #     # curr_err = (normX2 - np.sum(s**2)) / normX2
            
            # Update X[n]
            rn = v.shape[1]
            Xt.factors[n] = tl.reshape(v.T, (rn, SzX[n], rankX[n+1]))
            rankX[n] = rn
            #Xt.rank[n] = rn
            Xt.rank = rankX
 
 
            sum_rankX = np.append(sum_rankX,np.sum(rankX))
            
            if (prev_error is not None) and (abs(curr_err - prev_error) < tol) and (sum_rankX[cnt] == sum_rankX[cnt-1]) or (kiter == maxiters):
                stop_cnt += 1
            else:
                stop_cnt = 0
            
            
            
            # If the next core to be updated is not X[n-1], but e.g., X[n-2],
            # then Update X[n-1], and left-orthogonalize this core tensor
            u = u @ np.diag(s)
            
            if next_n <= (n-2):
                #print(f'Orthogonalize R2L U[{n-1}]')
                # Orthogonalization to Un-1 or Un
                # right orthogonalization to U[n-1]
                u = tl.reshape(u, (rankX[n-1], -1))
                u, uR = tl.qr(u.T, mode="reduced")
            
                rnm1 = u.shape[1]
                
                # no need adjust Xt[n-2]  because it will be updated in next iteration
                # if k_update == no_updates_R2L:
                Xt.factors[n-2] = tl.reshape(tl.reshape(Xt.factors[n-2], (-1, rankX[n-1])) @ uR.T, (rankX[n-2], SzX[n-2], -1))
                
                # Update X[n-1]
                Xt.factors[n-1] = tl.reshape(u.T, (-1, SzX[n-1], rn))
                rankX[n-1] = rnm1
                #Xt.rank[n-1] = rnm1
                Xt.rank = rankX
            else:
                # X[n-1] even need not to be updated, except the last run
                # if k_update == no_updates_R2L:
                Xt.factors[n-1] = tl.reshape(u, (-1, SzX[n-1], rn))
            
            # curr_err2 = tl.norm(tt_to_tensor(Xt) - X)**2/normX2
            # print(f'Modes {modes} {[curr_err, curr_err2]}')
            err = np.append(err,curr_err)

            prev_error = curr_err

            if (param["printitn"]>0) and (cnt % param["printitn"] == 0):  
                print(f"Iter {kiter}, Cores {modes}, Error {curr_err:.5f}, SumRank {sum_rankX[cnt]}")
                
            # CHECK Rank and size of Xt
            # for kc in range(N):
            #     print(f'Xt[{kc}].shape {Xt[kc].shape}')

            # print(f'rankXt {rankX}')
            # print(f'rankXt {Xt.rank}')
                
            prev_n = n
            if stop_cnt > max_stop_cnt:
                # if next_n == n-2:
                #     Xt.factors[n-2] = tl.reshape(tl.reshape(Xt.factors[n-2], (-1, rankX[n-1])) @ uR.T, (rankX[n-2], SzX[n-2], -1))
                # else:
                #     Xt.factors[n-1] = tl.reshape(u.T, (-1, SzX[n-1], rn))
                break 

        if stop_cnt > max_stop_cnt:
            break
        # end the main loop
    
    # if nargout >=2
    err = err[:cnt]
    #     sum_rankX = sum_rankX(1:cnt);
    #     output = struct('Fit',1-err,'NoIters',cnt,'Sum_rank',sum_rankX);
    # end
     
    
    def contract_update(Xt, X, Phi_left, n_prev, n_curr):
        # Xt is TT-tensor , X is a tensor of order-N (the same order of Xt
        #
        # Update the left- contraction tensors Phi_left
        #     n_prev:  previous mode
        #     n_curr:  current mode
        
        N = tl.ndim(X)
        # Update Phi_left only when go from left to right 1, 2, ...
        if (n_curr > 1) and (n_prev < n_curr): # Left-to-Right
            if n_curr == 2:
                Z = ttxt(Xt, X, 2, 'left') # left contraction of mode-n
                Z = tl.reshape(Z, (rankX[2], *SzX[2:]))
                Phi_left[2] = Z
        
            else:
                # Z = ttxt(Xt, X, n(1), 'left') # contraction except mode-n
                # Z = tl.reshape(Z, (rankX[n(1)], *SzX[n(1):]))
        
                if n_prev == 1:
                    Z = X
                else:
                    Z = Phi_left[n_prev]
                for kn3 in range(n_curr - n_prev):
                    Z = tl.reshape(Z, (rankX[n_prev + kn3] * SzX[n_prev + kn3], -1))
                    Z = tl.dot(tl.reshape(Xt.factors[n_prev + kn3], (rankX[n_prev + kn3] * SzX[n_prev + kn3], -1)).T, Z) # Rn x (In ... IN)
                    
                    Z = tl.reshape(Z, (rankX[n_prev + kn3 + 1], *SzX[n_prev + kn3 + 1:]))
                    Phi_left[n_prev + kn3 + 1] = Z
        return Phi_left
    
    
    def contraction_bothsides(n, Phi_left):
        # Compute the contraction matrix between X and Xt for both-side of
        # modes [n1, n2]
        #
        #  Phi_left[n] is left-contraction matrix between X and Xt at mode
        #  n
        #
        #  Phi_left[n] can be computed through Phi_left[n-1]
        
        progressive_mode = True
        if not progressive_mode:
            # Direct computation - expensive method
            Tn = ttxt(Xt, X, n, 'both') # contraction except mode-n
            
        else:
            
            if n[0] == 0:
                Tn = ttxt(Xt, X, n[1], 'right') # contraction except mode-n
            
            else:
                #  Phi-left is precomputed
                Z = Phi_left[n[0]]
                
                # right contraction
                for n2 in range(N-1, n[1], -1):
                    if n2 == N-1:
                        Z = tl.reshape(Z, (-1, SzX[N-1]))
                        Z = tl.dot(Z, Xt.factors[n2].T) # R2 x (I2 ... IN)
                    else:
                        Z = tl.reshape(Z, (-1, rankX[n2+1] * SzX[n2])) # In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = tl.dot(Z, tl.reshape(Xt.factors[n2], (-1, rankX[n2+1] * SzX[n2])).T) # R2 x (I2 ... IN)
                Tn = tl.reshape(Z, (rankX[n[0]], -1, rankX[n[1]+1]))
        return Tn, Phi_left
    
    return Xt, err
    



def ttmatrix_time_tt(Att,btt):
    """ TT-matrix times TT-tensor
    A is a matrix of size (I1 I2 ... I_N) x (J1 J2...J_N) represented in a TT/MPS format of order N with 
    row sizes (I1 I2 ... I_N)
    column sizes (J1 J2 ... J_N) 
    and ranks (1-R1-R2-...-R_(N-1)-1)

       I 1           I 2               I(N_1)              I(N)
        |             |                 |                   |       
    1--(u 1)--R(1)--(u 2)--R(2) ... --(u (N_1))-- R(N-1)--(u (N))-- 1
        |             |                 |                   |  
       J 1           J 2               J(N-1)              J(N)

    x is a vector of length ((J1 J2...J_N)), represented in a TT tensor of order N 
    with size (J1 J2 ... J_N)  and ranks (1-S1-S2-...-S_(N-1)-1)


    1--(v 1)--S(1)--(v 2)--S(2) ... --(v (N_1))-- S(N-1)--(v (N)))-- 1
        |             |                 |                   |  
       J 1           J 2               J(N-1)              J(N)

    """
 
    # TT-matrix times TT
    # A * x
    Arank = Att.rank
    brank = btt.rank
    N = len(btt)

    ctt = [None] * N  # Preallocate the list for Btt cores

    for k in range(N):
        # [R(k) I(k) J(k) R(k+1)]  x [Sk x J[k] x S(k+1)]
        # RkSk x Ik x R(k+1) S(k+1) 
        ctt[k] = einsum('rijt,sjp->rsitp',Att[k],btt[k]); 
        ctt[k] = ctt[k].reshape(Arank[k]*brank[k],-1,Arank[k+1]*brank[k+1])    

    ctt = TTTensor(ctt)

    return ctt


def ttmatrix_transpose(Att):
    # Tranpose a TT-matrix
    AT = [np.transpose(x,[0, 2,1,3]) for x in Att.factors]
    AT = TTMatrix(AT)

    return AT


def ttmatrix_AtA(Att):
    # Q = A^T A
    
    Arank = Att.rank
    N = len(Att)

    Qtt = [None] * N  # Preallocate the list for Qtt = Att^T Att 
    # Qtt [k] has size  Rk^2 x Jk x Jk x R(k+1)^2


    for k in range(N):
        # [R(k) I(k) J(k) R(k+1)]  x [R(k) I(k) J(k) R(k+1)]
        # Rk^2 x Jk x Jk x R(k+1)^2 
        Qtt[k] = einsum('rijt,sikp->rsjktp',Att[k],Att[k]); 
        Qtt[k] = Qtt[k].reshape(Arank[k]**2,Att[k].shape[2],Att[k].shape[2],Arank[k+1]**2)    

    Qtt = TTMatrix(Qtt)
    return Qtt


def xQx_left(Qtt,xtt,mode):
    # Q:  TT-matrix 
    # x : TT tensor
    # return contraction of the two TT-networks from core-1 to core-n (specified by mode)
    #   (Q(1)-x(1))-...(Q(mode-1)-x(mode-1))

    # Left multiplication x^T * Q * x upto mode n
    # TT-matrix times TT

    Qrank = Qtt.rank
    xrank = xtt.rank
    N = len(xtt)

    # Qxleft can be computed in progressive way,
    # Qxleft(n) = Qxleft(n-1) * Qx_[n]
    Qxleft = 1
    for k in range(mode):
        # print(k)
        # [Sk x J[k] x S(k+1)] x [R(k) J(k) J(k) R(k+1)]  x [Sk x J[k] x S(k+1)]
        # RkSk^2 x x R(k+1) S(k+1)^2

        Qx_k = einsum('rijt,piq,ujv->rputqv',Qtt[k],xtt[k],xtt[k]); 
        Qx_k = Qx_k.reshape(Qrank[k]*xrank[k]**2,Qrank[k+1]*xrank[k+1]**2)    
        
        if k == 0:
            Qxleft = Qx_k
        else:        
            Qxleft = Qxleft @ Qx_k
    
    return Qxleft


def xQx_right(Qtt,xtt,mode):
    # Q:  TT-matrix 
    # x : TT tensor
    # return contraction of the two TT-networks from core-n+1 to core-N (specified by mode)
    #   (Q(n+1)-x(n+1))-...(Q(N)-x(N))

    # Left multiplication x^T * Q * x upto mode n
    # TT-matrix times TT
    # Qxright can be computed in progressive way,
    # Qxright(n) = Qx_[n] *  Qxright(n+1)
    Qrank = Qtt.rank
    xrank = xtt.rank
    N = len(xtt)
    Qxright = 1
    for k in range(N-1,mode,-1):
        # print(k)
        # [Sk x J[k] x S(k+1)] x [R(k) J(k) J(k) R(k+1)]  x [Sk x J[k] x S(k+1)]
        # RkSk^2 x x R(k+1) S(k+1)^2

        Qx_k = einsum('rijt,piq,ujv->rputqv',Qtt[k],xtt[k],xtt[k]); 
        Qx_k = Qx_k.reshape(Qrank[k]*xrank[k]**2,Qrank[k+1]*xrank[k+1]**2)    
        
        if k == N-1:
            Qxright = Qx_k
        else:        
            Qxright = Qx_k @ Qxright
    
    return Qxright
 
def xQx_subset(Qtt,xtt,mode):
    # # Construct sub-network for the quadratic term x^T Q x = xn^T Qn xn
    # where xn = vec(x[n])
    Qrank = Qtt.rank
    xrank = xtt.rank
    N = len(xtt)
    Qxleft = xQx_left(Qtt,xtt,mode)
    Qxright = xQx_right(Qtt,xtt,mode)

    # x^T Q x = xn^T Pn xn
    if mode == 0:
        #  Qn - Qright
        Qxright = Qxright.reshape(Qrank[mode+1],xrank[mode+1]**2)
        Qn = einsum('rijs,sp->rijp',Qtt[mode],Qxright);  # s = 1
        Qn = Qn.reshape(Qn.shape[1],Qn.shape[2],xrank[mode+1],xrank[mode+1])
        Qn = np.transpose(Qn,[0,2,1,3])

    elif mode == N-1:
        
        # Qleft - Qn
        # Qxleft: R[k]^2 x S[k]^2
        # Q(N-1) # Rk^2 x Jk
        Qxleft = Qxleft.reshape(Qrank[mode],xrank[mode]**2)
        # Qtt[mode] : 
        Qn = einsum('rp,rijs->pijs',Qxleft,Qtt[mode]);  # s = 1
        Qn = Qn.reshape(xrank[mode],xrank[mode],Qn.shape[1],Qn.shape[2])
        Qn = np.transpose(Qn,[0,2,1,3])

    else:
        # x^T Q x = xn^T Pn xn
        # Qleft - Qn - Qright
        Qxright = Qxright.reshape(Qrank[mode+1],xrank[mode+1],xrank[mode+1])
        Qxleft = Qxleft.reshape(Qrank[mode],xrank[mode],xrank[mode])

        Qn = einsum('rpq,rijs,suv->pqijuv',Qxleft,Qtt[mode],Qxright); 
        Qn = np.transpose(Qn,[0,2,4,1,3,5])


    # x^T A^T A x =  xn(:)^T Qn xn(:)
    Qn = Qn.reshape(xtt[mode].size,xtt[mode].size)
    # bnf2 = xtt[mode].ravel().T @ Qn @ xtt[mode].ravel()  # ||A x||^2

    return Qn


def fx_left(ftt,xtt,mode):
    # f:  TT-tensor 
    # x : TT tensor
    # return contraction of the two TT-tensors from core-1 to core-n (specified by mode)
    #   (f(1)-x(1))-...(f(mode-1)-x(mode-1))

    # Left multiplication x^T * f upto mode n
    
    frank = ftt.rank
    xrank = xtt.rank
    N = len(xtt)

    # fxleft can be computed in progressive way,
    # fxleft(n) = fxleft(n-1) * fx_[n]
    fxleft = 1
    for k in range(mode):
        # print(k)
        
        fx_k = einsum('piq,uiv->puqv',ftt[k],xtt[k]); 
        fx_k = fx_k.reshape(frank[k]*xrank[k],frank[k+1]*xrank[k+1])    
        
        if k == 0:
            fxleft = fx_k
        else:        
            fxleft = fxleft @ fx_k
    
    return fxleft


def fx_right(ftt,xtt,mode):
    # f:  TT-tensor 
    # x : TT tensor
    # return contraction of the two TT-networks from core-n+1 to core-B (specified by mode)
    #   (f(n+1)-x(n+1))-...(f(N)-x(N))

    # Left multiplication x^T * Q * x upto mode n
    # TT-matrix times TT
    # Qxright can be computed in progressive way,
    # Qxright(n) = Qx_[n] *  Qxright(n+1)
    frank = ftt.rank
    xrank = xtt.rank
    N = len(xtt)
    fxright = 1
    for k in range(N-1,mode,-1):
        # print(k) 
        fx_k = einsum('piq,uiv->puqv',ftt[k],xtt[k]); 
        fx_k = fx_k.reshape(frank[k]*xrank[k],frank[k+1]*xrank[k+1])     
        
        if k == N-1:
            fxright = fx_k
        else:        
            fxright = fx_k @ fxright
    
    return fxright
 
def fx_subset(ftt,xtt,mode):
    # # Construct sub-network for product x^T f = xn^T Qn xn
    # where xn = vec(x[n])
    frank = ftt.rank
    xrank = xtt.rank
    N = len(xtt)
    fxleft = fx_left(ftt,xtt,mode)
    fxright = fx_right(ftt,xtt,mode)

    # x^T f = xn^T fn
    if mode == 0:
        #  fn - fright
        fxright = fxright.reshape(frank[mode+1],xrank[mode+1])
        fn = einsum('rjs,sp->rjp',ftt[mode],fxright);  # s = 1        
        
    elif mode == N-1:
        # fleft - fn
        # fxleft: R[k] x S[k]
        # f(N-1) # Rk x Jk
        fxleft = fxleft.reshape(frank[mode],xrank[mode])
        # ftt[mode] : 
        fn = einsum('rp,rjs->pjs',fxleft,ftt[mode]);  # s = 1

    else:
        # x^T Q x = xn^T Pn xn
        # Qleft - Qn - Qright
        fxright = fxright.reshape(frank[mode+1],xrank[mode+1])
        fxleft = fxleft.reshape(frank[mode],xrank[mode])

        fn = einsum('rp,ris,su->piu',fxleft,ftt[mode],fxright);         

    # x^T A^T A x =  xn(:)^T Qn xn(:)
    fn = fn.reshape(xtt[mode].size)
    # bnf2 = xtt[mode].ravel().T @ Qn @ xtt[mode].ravel()  # ||A x||^2

    return fn





# For complex valued tensor


def tt_nestedtk2_complex_tensor(Y,approx_bound,precision = 1e-9):
    # Nested TUCKER-2 for TT decomposition 
    # Y is approximated by a TT -tensor 
    # Ytt = Uleft_1 *  G1 * Uright_1 (Tucker-2 model)
    # G{k-1} = Uleft_k *  Gk * Uright_k 
    #
    # 
    # Anh-Huy Phan
    # 
    # approx_bound = noiselevel**2 * Y.size
    # precision = 1e-9 # small number added to the approximation error bound 
    normY2 = tl.norm(Y)**2
    szY = Y.shape
    N = Y.ndim
    rankTTx = np.zeros(N+1, dtype = 'int')
    rankTTx[0] = 1
    rankTTx[N] = 1

    
    Yk = Y   
    
    # Factors of the nested TT 
    Factors = [None] * N
    maxiters = 1000
    
    for k in range(np.array((N-N%2)/2,int)):
        # Layer-K 
        print(f'Perform the {k+1}-th Tucker-2 decomposition')
        # approximation error = \|Y - Ul(1:K)*Yxk*Ur(K:1)\|_F^2
        #  = \|Y\|_F^2 + \|Yxk\|_F^2 - 2<Yk, Yxk>
        #  = \|Yk - Yxk\|_F^2 - ||Yxk||_F^2 +\|Y\|_F^2 <= delta^2
        # 
        # \|Yk - Yxk\|_F^2 <= delta^2 - \|Y\|_F^2 +||Yk||_F^2
         
        if ((N%2)==0) and (k == (np.array((N-N%2)/2,int)-1)):
            # For the last level when N is an even number
            Yk = Yk.reshape(rankTTx[k]*szY[k],rankTTx[N-k]*szY[N-1-k])
            normYk = tl.norm(Yk)**2
            approx_bound_n = approx_bound - normY2 + normYk
            approx_bound_n = approx_bound_n + precision
         
            Ulk, s, Urk, approx_error,_ = lowrank_matrix_approx(Yk, approx_bound_n, exacterrorbound=False)
            Ulk = Ulk * s       
        
        else:
            # Tucker-2 decomposition 
            Yk = Yk.reshape(rankTTx[k]*szY[k],-1,rankTTx[N-k]*szY[N-1-k])
            normYk = tl.norm(Yk)**2
            approx_bound_n = approx_bound - normY2 + normYk
            approx_bound_n = approx_bound_n+precision
            noiselevel_n = np.sqrt(approx_bound_n/Yk.size)
                        
            # initialization
            Ulk, Urk, err,_,_ = tucker2_truncatedhosvd_init_complex_tensor(Yk,approx_bound_n)
            # Tucker2 decomposition with given approximation error 
            Ulk,Yknew,Urk,approx_error,noparams,rankR = fast_tucker2_denoising_complex_tensor(Yk, [Ulk, Urk], maxiters, 1e-7, noiselevel_n, exacterrorbound=False, precision=1e-7, verbose=True) 
            
        
        # print(f'Approximation Error {approx_error[-1]} | Predefined Bound {approx_bound}')

        approx_error = normY2 - normYk + approx_error
        # Update ranks
        rankTTx[1+k] = Ulk.shape[1]
        rankTTx[N-1-k] = Urk.shape[1]
    
        # Update the left factor
        Factors[k] = Ulk.reshape(rankTTx[k],szY[k],rankTTx[k+1])
        # Update the right factor
        Factors[N-1-k] = Urk.T.reshape(rankTTx[N-1-k],szY[N-1-k],rankTTx[N-k]) 
        Yk  = Yknew
    
    if (N%2)==1:
        # core tensor in 
        k =np.array((N-N%2)/2,int)
        Factors[k] = Yk.reshape(rankTTx[k],szY[k],rankTTx[k+1])
    
    # for n in range(N):     
    #     Factors[n] = Factors[n].reshape(rankTTx[n],szY[n],rankTTx[n+1])
    
    Yttnested = TTTensor(Factors)
    # print(Yttnested)
    # approx_error = tl.norm(Y - tt_to_tensor(Yttnested))**2
    
    print(f'Approximation Error {approx_error} | Predefined Bound {approx_bound}')
    return Yttnested, approx_error




from tensorly.tenalg.svd import svd_interface
from tensorly.tt_tensor import validate_tt_rank, TTTensor
def tt_svd(input_tensor, rank, svd="truncated_svd", verbose=False):
    """TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
              order-3 tensors of the TT decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    # rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        # U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method=svd) 

        U, S, V = tl.truncated_svd(unfolding, n_eigenvecs=current_rank) # fix error for complex matrix

        # U, S, V = U[:, :current_rank], S[:current_rank], V[:current_rank, :]


        rank[k + 1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if verbose is True:
            print(
                "TT factor " + str(k) + " computed with shape " + str(factors[k].shape)
            )

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    if verbose is True:
        print(
            "TT factor "
            + str(n_dim - 1)
            + " computed with shape "
            + str(factors[n_dim - 1].shape)
        )

    return TTTensor(factors)
