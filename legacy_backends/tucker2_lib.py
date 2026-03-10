import tensorly as tl
import numpy as np
from numpy.linalg import norm
import scipy
from scipy.linalg import svd
from scipy.io import loadmat
from scipy.sparse.linalg import eigs

"""
tucker_orthogonalization(X) : orthogonalization of factor matrices 
minrank_boundedtrace(Q, delta, exactbound=True, precision=1e-5):  
    min      rank(A)
    subject to   g(A) = trace(A'*Q*A) >=bound
                 A'*A = I  (A: orthogonal matrix)

fast_tucker2_denoising(X, init, maxiters, tol, noiselevel, exacterrorbound=None, precision=None, verbose = False)

tucker2_truncatedhosvd_init(Y,approx_bound): Initialization for Tucker decomposition with error bound constraint
    ||X - Y||_F^2 <= approx_bound
    X is a Tucker2 model

tucker_denoising(X, init, maxiters, tol, noiselevel, decomposemodes, exacterrorbound=None, precision=None, verbose=None)
tucker_truncatedhosvd_init(X,approx_bound,decomposemodes)

Anh-Huy Phan

"""

def minrank_boundedtrace(Q, delta, exactbound=True, precision=1e-5):
    # Solve the mininimization problem
    #    min      rank(A)
    # subject to   g(A) = trace(A'*Q*A) >=bound
    #              A'*A = I  (A: orthogonal matrix)
    # 
    # PHAN ANH-HUY
    #  
    if np.linalg.norm(Q, 'fro')**2 < delta:
        return [], [], []

    I = Q.shape[0]
    lambda_diag, U = np.linalg.eigh(Q)
    il = np.argsort(lambda_diag)[::-1]
    lambda_diag = lambda_diag[il]
    U = U[:, il]

    sum_lambda = np.cumsum(lambda_diag)
    R = np.argmax(sum_lambda >= delta) + 1
 
    if R == 0 or R == I:
        R = np.argmax(sum_lambda >= delta * (1 - precision)) + 1

    if R == 0:
        return [], [], []

    if R == I:
        X = U
        trace_val = sum(lambda_diag)
        return X, R, trace_val

    trace_val = sum(lambda_diag[:R])
    if not exactbound or abs(trace_val - delta) <= delta * precision:
        X = U[:, :R]
        return X, R, trace_val

    # for the case : exact bound 
    selset = selection_strategy(lambda_diag, R, delta)

    Xi1 = np.eye(I)[:, selset]
    fXi1 = sum(lambda_diag[selset])

    swapcomp = np.where(selset > R)[0][-1]
    setd = np.setdiff1d(np.arange(1, R+1), selset)
    swapcomp1 = setd[0]

    k = [swapcomp1, selset[swapcomp]]
    lambda_k = lambda_diag[k]
    gamma_z = 2 * (delta - fXi1) / (lambda_k[0] - lambda_k[1]) - 1
    z_x = (np.pi - np.arccos(gamma_z)) / 4
    w_x = np.array([np.cos(z_x), np.sin(z_x)])

    Xi2 = Xi1.copy()
    Xi2[k, :] = Xi2[k, :] - 2 * np.outer(w_x, w_x @ Xi1[k, :])
    trace_val = np.trace(Xi2.T @ np.diag(lambda_diag) @ Xi2)
    X = U @ Xi2

    return X, R, trace_val


def selection_strategy(lambda_, R, delta):
    I = len(lambda_)
    bagset = list(range(I-R+1, I+1))  # note that sum(lambda_[bagset]) < delta
    selset = list(range(1, R+1))
    if R == 1:
        bagset2 = bagset.copy()
        K = 0
    else:
        #print(lambda_.shape)
        for remnum in range(R-1, 0, -1):
            # replace the 1st entry
            bagset2 = bagset.copy()
            bagset2[:remnum] = selset[:remnum]
            #print(bagset2)
            if np.sum(lambda_[bagset2]) <= delta:
                break
            # end if
        # end for loop
        K = remnum
    # Refine the selected set of indices
    bagset3 = bagset2.copy()
    refix1 = K+1
    refix2 = bagset2[K]-1
    exset = list(range(refix1, refix2+1))
    for bagix in range(K, len(bagset2)):
        # find an entry within the interval refix:bagset3[bagix]-1
        # to replace the entry #bagix
        ix_options = [ix for ix in exset if lambda_[ix-1] <= delta - np.sum(np.array(lambda_)[bagset3[:bagix] + bagset3[bagix+1:]])]
        if ix_options:
            ix = ix_options[0]
            bagset3[bagix] = ix
            refix1 = ix + 1
            exset = list(range(refix1, refix2+1))
    selset = bagset3
    return selset

    
def fast_tucker2_denoising(X, init, maxiters, tol, noiselevel, exacterrorbound=None, precision=None, verbose=None):
    """
    Simple fast Tucker-2 decomposition for denoising problem
    min #param(G) s.t. |X - [[G;U1 U3]] |_F^2 <= noiselevel^2 * numel(X)
    PHAN ANH-HUY
    init = [U1_init, U3_init]: initialization for the factor matrices U1 and U3 
    """
    if exacterrorbound is None:
        exacterrorbound = True
    if precision is None:
        precision = 1e-8
    if verbose is None:
        verbose = False
        
    SzX = X.shape
    # Set up and error checking on initial guess for U.
    if isinstance(init, list):
        Uinit = init
        if len(Uinit) != 2:
            raise ValueError('init does not have %d cells' % 2)
        if Uinit[0] is not None and Uinit[0].shape[0] != SzX[0]:
            raise ValueError('init{%d} is the wrong size' % 1)
        if Uinit[1] is not None and Uinit[1].shape[0] != SzX[2]:
            raise ValueError('init{%d} is the wrong size' % 2)
    else:
        Uinit = [None, None]
        if init == 'random':
            Uinit[1] = np.random.rand(SzX[2])
        elif init in ['nvecs', 'eigs']:
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (1 <= n <= N).
            print('  Computing leading e-vectors for factor %d.' % 2)
            Uinit[1] = tl.tenalg.svd_interface(tl.base.unfold(X,mode=2), n_eigenvecs=SzX[2])[0]
        else:
            raise ValueError('The selected initialization method is not supported')
    # Set up for iterations - initializing U and the fit.
    U1 = Uinit[0] if Uinit[0] is not None else np.eye(SzX[0])
    U3 = Uinit[1] if Uinit[1] is not None else np.eye(SzX[2])

    if verbose:
        print('\nTucker-2 for denoising:\n')
        
    normX2 = norm(X.flatten())**2
    
    if noiselevel >= 0:
        # Minimize rank or number of parameters with error bound constraints
        # min #param(Xhat) s.t. |X - [[G;U1 U3]] |_F^2 <= noiselevel^2 * numel(X)
        approx_bound = noiselevel**2 * np.prod(SzX)
        accuracy = normX2 - approx_bound
        rank_minimization = True
    else:
        # Minimize approximation error
        # min |X - [[G;U1 U3]] |_F^2
        rank_minimization = False
        accuracy = normX2
        
    # Permute
    permuteX = SzX[2] > SzX[0] * 1.5
    if permuteX:
        SzX = (SzX[2], SzX[1], SzX[0])
        X = np.transpose(X, (2, 1, 0))
        U1, U3 = U3, U1

    
    # Reshape X
    X1 = X.reshape((X.shape[0], -1))
    X3 = X.reshape((-1, X.shape[2]))
    
    # Get the sizes of U1 and U3
    R = [U1.shape[1], U3.shape[1]]
    
    # Compute T and U2
    print(f'Ranks {R}')
    
    T = U1.T @ X1
    T = T.reshape((-1, X.shape[2]))
    U2 = T @ U3
    U2 = U2.reshape((R[0], -1, R[1]))
    normU2_ = norm(U2.reshape(-1))**2
    
    # The loop updates U1 and U3, while U2 is computed after the iteration
    if rank_minimization:
        constrval = np.zeros(maxiters)
        rankR = np.zeros((maxiters, 2))
    objval = np.zeros(maxiters)
    
    for kiter in range(maxiters):
        # Update U1
        T = X3 @ U3
        T = T.reshape((X.shape[0], -1))
        Q = T @ T.T
        
        u1, r1, trace_val = minrank_boundedtrace(Q, accuracy, exacterrorbound, precision)
        
        if u1 is not None:
            U1 = u1
            R[0] = r1
            normU2_ = trace_val
        
        # Update U3
        T = U1.T @ X1
        T = T.reshape((-1, X.shape[2]))
        Q = T.T @ T
        
        u3, r3, trace_val = minrank_boundedtrace(Q, accuracy, exacterrorbound, precision)
        
        if u3 is not None:
            U3 = u3
            R[1] = r3
            normU2_ = trace_val
        
        # Stop if converged
        if rank_minimization:
            # for the rank minimization problem 
            constrval[kiter] = normU2_
            objval[kiter] = X.shape[0] * R[0] + X.shape[2] * R[1] + X.shape[1] * R[0] * R[1]
            rankR[kiter, :] = R
        else:
            objval[kiter] = normU2_
        
        if verbose:
            if rank_minimization:
                print('%d  | (R1,R2) (%d, %d) |  #params %d | approx %.4g  | approx.bound %.4g' % (kiter, R[0], R[1], objval[kiter],(normX2-constrval[kiter])/normX2,approx_bound/normX2))
            else:
                # minimization of approximation error 
                print('%d  | (R1,R2) (%d, %d) |  #params %d'  % (kiter, R[0], R[1], objval[kiter]))
        
        if kiter > 0:
            isconverge = (abs(objval[kiter] - objval[kiter-1]) <= tol)
            if rank_minimization:
                isconverge = isconverge and (abs(constrval[kiter] - constrval[kiter-1]) <= tol)
            if isconverge:
                break
    
    # Compute U2
    U2 = T @ U3
    U2 = U2.reshape((R[0], -1, R[1]))
    
    # Remove extra values
    objval = objval[:kiter+1]
    if rank_minimization:
        constrval = constrval[:kiter+1]  # ||X - Un Phi(uk)||_F^2
        rankR = rankR[:kiter+1, :]
        aprxerror = normX2 - constrval
        noparams = objval
    else:
        noparams = X.shape[0] * R[0] + X.shape[2] * R[1] + X.shape[1] * R[0] * R[1]
        aprxerror = normX2 - objval
    
    # Permute if needed
    if permuteX:
        U2 = np.transpose(U2, (2, 1, 0))
        U1, U3 = U3, U1

    return U1,U2,U3,aprxerror,noparams,rankR

def tucker2_truncatedhosvd_init(Y,approx_bound):
    normY = norm(Y.reshape(-1))
    # min rank(G) s.t. |X - [[G;U1 U3]] |_F^2 < noiselevel^2 * numel(X)
    # apx_bound = 0.01*normY**2
    # noiselevel = np.sqrt(apx_bound/Y.size)
    
    szY = Y.shape
    U1x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=0), n_eigenvecs=szY[0])[0]
    U3x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=2), n_eigenvecs=szY[2])[0]
    U2 = tl.tenalg.multi_mode_dot(Y,[U1x, U3x],modes = [0, 2], transpose=True)
    
    # find the smallest truncated model such that |U2(1:R1,:1:R2)|_F^2 <= apx_bound
    U2p = np.sum(U2**2, axis=1)
    U2p = np.cumsum(np.cumsum(U2p, axis=0), axis=1)
    # U2p[U2p < (normY**2 - apx_bound)] = np.nan
    
    R1 = np.arange(1, szY[0] + 1)
    R3 = np.arange(1, szY[2] + 1)
    R1x, R3x = np.meshgrid(R3, R1)
    numpararam = szY[0] * R1x + szY[2] * R3x + szY[1] * R1x * R3x
    
    # Find the indices where U2p >= (normY**2 - apx_bound)
    indices = np.where(U2p >= (normY**2 - approx_bound))
    
    # Extract the corresponding elements from numpararam
    numpararam_filtered = numpararam[indices]
    
    # Find the indices of the minimum element in B_filtered
    imin_index = np.argmin(numpararam_filtered)
    
    # print(imin_index)
    # # Convert the flattened index to the original shape
    # best_noparams = np.unravel_index(imin_index, B.shape)
    
    Rx = [R1[indices[0][imin_index]],R3[indices[1][imin_index]]]
    
    # print(Rx)
    # numpararam[U2p >= (normY**2 - apx_bound)] = np.nan
    # best_noparams, irr = np.nanmin(numpararam), np.nanargmin(numpararam)
    # irr = np.unravel_index(irr, U2p.shape)
    
    # rank TK2
    # Rx = [R1[irr[0]], R3[irr[1]]]
    
    # U2 = U2[0:Rx[0], :, 0:Rx[1]]
    # U2 = ttm(tensor(Y), [U1[:, 0:Rx[0]], U3[:, 0:Rx[1]]], [1, 3], 't')
    
    # approximation error 
    err = normY**2 - U2p[Rx[0]-1, Rx[1]-1]
    
    U1s, U3s = U1x[:, 0:Rx[0]], U3x[:, 0:Rx[1]]
 
    return U1s, U3s, err, U1x, U3x

def tucker2_to_tensor(U1,U2,U3):
    return np.einsum('ir,rjs,ks-> ijk',U1,U2,U3)


# Execute the Fast-Tucker decomposition with error bound constraint
# Truncated and Iterative 
#  see fast_tucker2_denoising
def exec_fast_tucker2_denoising(Y, Uinit, maxiters=100, tol=1e-5, sigma_noise=None, exactbound=False, boundprecision=1e-8, no_searches=10, notests_search=10, verbose=False, traceresult=False):
    szY = Y.shape
    if Uinit is None:
        Uinit = 'hosvd'
    
    # approximation bound :  |Y - Yx|_F^2 <= sigma_noise^2 * no_data_samples
    rank_minimization = sigma_noise is not None
    if rank_minimization:
        apx_bound = sigma_noise**2 * Y.size
    
    # Seek a good initial using truncated HOSVD
    U1x = None
    U3x = None
    giveninit = False

    if isinstance(Uinit, str) and Uinit == 'hosvd' and rank_minimization:
        # left and right factor matrices    
        # core tensor within HOSVD
        
        U1s, U3s, err, U1x, U3x = tucker2_truncatedhosvd_init(Y,apx_bound)
        Rx = [U1s.shape[1], U3s.shape[1]]
        
    elif isinstance(Uinit, list):
        giveninit = True
        U1s, U3s = Uinit[0], Uinit[1]
        Rx = [U1s.shape[1], U3s.shape[1]]
        
        err = None
    else:
        U1s, U3s = None, None
        Rx = [szY[0], szY[2]]
        err = 0
    
    best_noparams = szY[0] * Rx[0] + szY[2] * Rx[1] + szY[1] * Rx[0] * Rx[1]
    bestNoparams = best_noparams
    bestErr = err
    # print('%d    %d \n' % (best_noparams, err[-1]))
    
    # prepare for multi search
    if no_searches >= 1 and U1x is None:
        U1x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=0), n_eigenvecs=Rx[0])[0]
        U3x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=2), n_eigenvecs=Rx[1])[0]
        
    #Y = Y.tondarray()
    # Search and Refine
    Rxs = Rx
    
    if traceresult:
        errs2 = err
        noparams_ = best_noparams
        errs_ = err
        rankR_ = Rx
        rankRs = Rx

    # print(f'no_searches {no_searches}')
    # print(f'notests_search {notests_search}')
    
    # Main Loop over the number of searches
    for ki in range(no_searches):
        Rx = Rxs
         
        best_noparams_old = best_noparams
        #if verbose:
        #print('Run %d ' % ki)
        err_ = []
        errs = []

        # Loop over the number of tests per search
        for krun in range(notests_search):
            print(f'Run {ki} - {krun}')
            if krun == 0: # svd
                if Rx[0] <= U1x.shape[1]:
                    U1i = U1x[:, 0:Rx[0]]
                else:
                    U1i, _ = np.linalg.qr(np.hstack((U1x, np.random.randn(szY[0], U1x.shape[1] - Rx[0]))),'reduced')
                if Rx[1] <= U3x.shape[1]:
                    U3i = U3x[:, 0:Rx[1]]
                else:
                    U3i, _ = np.linalg.qr(np.hstack((U3x, np.random.randn(szY[2], U3x.shape[1] - Rx[1]))),'reduced')
            else:
                if giveninit and (ki == 0):                    
                    Uinit[0], Uinit[1] = U1s, U3s
                else:
                    typeinit = np.random.randint(1, 4)
                    # print(f'init type {typeinit}')
                    
                    if typeinit == 1:
                        u = scipy.linalg.orth(np.random.rand(U1s.shape[0], 1))
                        U1i = scipy.linalg.orth(U1s - 2 * u @ (u.T @ U1s))
                        u = scipy.linalg.orth(np.random.rand(U3s.shape[0], 1))
                        U3i = scipy.linalg.orth(U3s - 2 * u @ (u.T @ U3s))
                    elif typeinit == 2:
                        U1i = scipy.linalg.orth(U1s + np.random.randn(*U1s.shape) * 0.01)
                        U3i = scipy.linalg.orth(U3s + np.random.randn(*U3s.shape) * 0.01)
                    elif typeinit == 3:
                        U1i = scipy.linalg.orth(np.random.randn(szY[0], Rx[0]))
                        U3i = scipy.linalg.orth(np.random.randn(szY[2], Rx[1]))
            
            # print(f'Size U1i {U1i.shape}, Size U3i {U3i.shape}')
            
            # Call the fast Tucker-2 denoising function
            # print(f'Noise level {sigma_noise}')
            U1, U2, U3, err, noparams, rankR = fast_tucker2_denoising(Y, [U1i, U3i], maxiters, tol, sigma_noise, exacterrorbound=False, precision=boundprecision, verbose=verbose) 


            # Update the ranks
            if (err[-1]<=apx_bound):
                Rx = [U2.shape[0], U2.shape[2]]
            
            if verbose:
                print('%d    %d \n' % (noparams[-1], err[-1]))
            
            # Update the best solution
            if (err[-1]<=apx_bound) and ((best_noparams > noparams[-1]) or ((best_noparams == noparams[-1]) and (len(errs)>=1) and (errs > err[-1]))):
                U1s, U2s, U3s = U1, U2, U3
                Rxs = Rx
                errs = err[-1] # update the final approximation error 
                errs2 = err
                rankRs = rankR
                if best_noparams > noparams[-1]:
                    best_noparams = noparams[-1]  # update the best model with the smallest number of parameters
                    if (giveninit and (krun >= 1)) or (not giveninit and (krun >= 0)):                         
                        break
            
            # Append the error
            err_ = np.append(err_, err[-1])
             
            # Stop if there is no improvement
            if (krun >= 2) and np.all(np.abs(err_ - err[-1]) < 1e-5):
                break

            # print(f'End of Run {ki} - {krun}')
            
        # Append the errors and number of parameters for output
        if traceresult:
            if len(errs2) > 0:
                errs = np.append(errs, errs2)
            noparams = np.append(noparams, noparams)
            rankR_ = np.vstack((rankR_, rankRs))
        # enf of traceresult
        
        # Stop if there is no improvement
        if best_noparams == best_noparams_old:
            break
        bestNoparams = np.append(bestNoparams, best_noparams)
        bestErr = np.append(bestErr, err_)

    # end of ki loop
    
    if exactbound:
        U1, U2, U3, err, noparams, rankR = fast_tucker2_denoising(Y, [U1s, U3s], 100, tol, sigma_noise, exacterrorbound=True, precision=1e-7, verbose=None) 

    else:
        U1, U2, U3, err, noparams, rankR = fast_tucker2_denoising(Y, [U1s, U3s], 100, tol, sigma_noise, exacterrorbound=False, precision=1e-8, verbose=None) 

    bestNoparams = np.append(bestNoparams, noparams)

    bestErr = np.append(bestErr, err)
     
    
    if traceresult:  # concatenate errors and no_parameters for output
        errs_ = np.append(errs_,err)
        noparams_ = np.append(noparams_,noparams)
        rankR_ = np.append(rankR_,rankR)
     
    return  U1,U2,U3,bestErr,bestNoparams,errs_,noparams_,rankR_





    
def tucker_denoising(X, init, maxiters, tol, noiselevel, decomposemodes, exacterrorbound=None, precision=None, verbose=None):
    """
    Tucker decomposition for denoising problem
    min #param(G) s.t. |X - [[G;U1, U2, ...,  UN]] |_F^2 <= noiselevel^2 * numel(X)
    PHAN ANH-HUY
    init = [U1_init, .., UN_init]: initialization for the factor matrices Un
    """
    SzX = X.shape
    SzX = np.array(SzX)

    N = X.ndim
    if exacterrorbound is None:
        exacterrorbound = False
    if precision is None:
        precision = 1e-8
    if verbose is None:
        verbose = False

    if decomposemodes is None:
        decomposemodes = np.arange(N)

    numfactors = len(decomposemodes)
        
    
    # Set up and error checking on initial guess for U.
    if isinstance(init, list):
        Uinit = init
        if len(Uinit) != numfactors:
            raise ValueError('init does not have %d cells' % numfactors)

        for n in range(numfactors):
            if Uinit[n] is not None and Uinit[n].shape[0] != SzX[decomposemodes[n]]:
                raise ValueError('init{%d} is the wrong size' % n)
        
    else:
        Uinit = [None] * N
        if init == 'random':
            for n in range(numfactors):
                Uinit[n] = scipy.linalg.orth(np.random.rand(SzX[decomposemodes[n]]))
                
        elif init in ['nvecs', 'eigs']:
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (1 <= n <= N).
            for n in range(numfactors):
                print('  Computing leading e-vectors for factor %d.' % n)
                Uinit[n] = tl.tenalg.svd_interface(tl.base.unfold(X,mode=decomposemodes[n]), n_eigenvecs=SzX[decomposemodes[n]])[0]
        else:
            raise ValueError('The selected initialization method is not supported')

    
    # Set up for iterations - initializing U and the fit.
    U = Uinit
    for n in range(numfactors):
        U[n] = Uinit[n] if Uinit[n] is not None else np.eye(SzX[decomposemodes[n]])

    if verbose:
        print('\nTucker decomposition with error bound constraint:\n')
        
    normX2 = norm(X.flatten())**2
    
    if noiselevel >= 0:
        # Minimize rank or number of parameters with error bound constraints
        # min #param(Xhat) s.t. |X - [[G;U1... UN]] |_F^2 <= noiselevel^2 * numel(X)
        approx_bound = noiselevel**2 * np.prod(SzX)
        accuracy = normX2 - approx_bound
        rank_minimization = True
    else:
        # Minimize approximation error
        # min |X - [[G;U1... UN]] |_F^2
        rank_minimization = False
        accuracy = normX2
      
    # Get the sizes of U1 and U3
    R = [U[n].shape[1] for n in range(numfactors)]
    
    # Compute core tensor G
    print(f'Ranks {R}')

    G = tl.tenalg.multi_mode_dot(X,U,modes = decomposemodes, transpose=True)
    normG_ = norm(G.reshape(-1))**2
    numparamTK = lambda R : np.sum(SzX[decomposemodes] * R) + np.prod(SzX)/np.prod(SzX[decomposemodes])*np.prod(R)
    
    # The loop updates Un 
    if rank_minimization:
        constrval = np.zeros(maxiters)
        constrval[0] = normX2-normG_
        # print(f'Approximation error {constrval[0]} bound {approx_bound}')
        rankR = np.zeros((maxiters, numfactors))
    objval = np.zeros(maxiters)

    # [print(f'U[{k}] size {U[k].shape}') for k in range(numfactors)]
    
    
    for kiter in range(maxiters):
        for n in range(numfactors):

            # print(f'Update U[{n}]')
            
            allbutn = np.delete(decomposemodes,n)   
           
            U_butn = [U[k] for k in np.delete(np.arange(numfactors),n)]

                        
            # Update Un
            T = tl.tenalg.multi_mode_dot(X,U_butn,modes = allbutn, transpose=True)
            T = tl.unfold(T,mode = decomposemodes[n])
            Q = T @ T.T
            
            u, r, trace_val = minrank_boundedtrace(Q, accuracy, exactbound = False, precision = 1e-8)

            # check min rank with bounded trace 
            # print(f'Rank new {r}, u.shape {u.shape}, {trace_val} {accuracy}') 
            if u is not None:
                U[n] = u
                R[n] = r
                normG_ = trace_val
             
        # Stop if converged
        if rank_minimization:
            # for the rank minimization problem 
            constrval[kiter] = normG_
            objval[kiter] = numparamTK(R)
            rankR[kiter, :] = R
        else:
            objval[kiter] = normG_
        
        if verbose:
            if rank_minimization:
                print('%d  | #params %d | approx %.4g  | approx.bound %.4g' % (kiter, objval[kiter],(normX2-constrval[kiter])/normX2,approx_bound/normX2))
            else:
                # minimization of approximation error 
                print('%d  | #params %d'  % (kiter, objval[kiter]))
        
        if kiter > 0:
            isconverge = (abs(objval[kiter] - objval[kiter-1]) <= tol)
            if rank_minimization:
                isconverge = isconverge and (abs(constrval[kiter] - constrval[kiter-1]) <= tol)
            if isconverge:
                break
    
    # Compute core tensor
    G = tl.tenalg.multi_mode_dot(X,U,modes = decomposemodes, transpose=True)
    
    # Remove extra values
    objval = objval[:kiter+1]
    if rank_minimization:
        constrval = constrval[:kiter+1]  # ||X - Un Phi(uk)||_F^2
        rankR = rankR[:kiter+1, :]
        aprxerror = normX2 - constrval
        noparams = objval
    else:
        noparams = numparamTK(R)
        aprxerror = normX2 - objval 
    
    return U,G,aprxerror,noparams,rankR



def tucker_truncatedhosvd_init(X,approx_bound,decomposemodes):
    SzX = X.shape
    SzX = np.array(SzX)
    N = X.ndim
     
    if decomposemodes is None:
        decomposemodes = np.arange(N)
    
    numfactors = len(decomposemodes)
    normX = norm(X.reshape(-1))
    # min rank(G) s.t. |X - [[G;U1... UN]] |_F^2 < noiselevel^2 * numel(X)
    # apx_bound = 0.01*normY**2
    # noiselevel = np.sqrt(apx_bound/Y.size)

    # Factor matrices of the HOSVD, i.e., matrix of singular vectors of mode-n unfolding X_n
    Ux = [None] * numfactors
    for n in range(numfactors):
        Ux[n] = tl.tenalg.svd_interface(tl.base.unfold(X,mode=decomposemodes[n]), n_eigenvecs=SzX[decomposemodes[n]])[0]
    
    # core tensor
    G = tl.tenalg.multi_mode_dot(X,Ux,modes = decomposemodes, transpose=True)
     
    # find the smallest truncated model such that |G(1:R1,...,1:RN)|_F^2 <= apx_bound
    # ||X - Xhat||_F^2 = ||X||_F^2 - ||G||_F^2
    G2 = G**2
    if N > numfactors:
        # sum G**2 over all non-decomposed modes
        nondecomposemodes = np.delete(np.arange(N),decomposemodes)
        for n in nondecomposemodes[::-1]:
            G2 = np.sum(G2, axis=n)
    # print(G2.shape)
    # cum-sum over decomposed modes 
    for n in range(numfactors):
        G2 = np.cumsum(G2, axis=numfactors-n-1)

    # meshgrid of rank combination 
    Rx = [np.arange(n)+1 for n in SzX[decomposemodes]]
    Rnx = np.meshgrid(*Rx,indexing = 'ij')

    # number of model (Tucker) parameters)
    # number of parameters of the core tensors
    numpararam = np.prod(SzX)/np.prod(SzX[decomposemodes])
    for n in range(numfactors):
        numpararam = numpararam * Rnx[n]
    # number of parameters of the factor matrices
    for n in range(numfactors):
        numpararam = numpararam + Rnx[n]*SzX[decomposemodes[n]]
    
    # Find the indices where ||G||_F^2 >= (normY**2 - apx_bound)
    indices = np.where(G2 >= (normX**2 - approx_bound))
    
    # Extract the corresponding elements from numpararam
    numpararam_filtered = numpararam[indices]
    
    # Find the indices of the minimum element in numpararam_filtered
    imin_index = np.argmin(numpararam_filtered)
    
    # Truncated Ranks 
    Rs = [Rx[n][indices[n][imin_index]]-1 for n in range(numfactors)]
    Rs = np.array(Rs)
    
    # approximation error of the truncated model 
    err = normX**2 - G2[tuple(Rs)]
    print(f'{err} bound {approx_bound}')
    
    Us = [Ux[n][:, 0:Rs[n]+1] for n in range(numfactors)]
    
    return Us, Rs, err, Ux





# For complex valued tensor

def fast_tucker2_denoising_complex_tensor(X, init, maxiters, tol, noiselevel, exacterrorbound=None, precision=None, verbose=None):
    """
    Simple fast Tucker-2 decomposition for denoising problem
    min #param(G) s.t. |X - [[G;U1 U3]] |_F^2 <= noiselevel^2 * numel(X)
    PHAN ANH-HUY
    init = [U1_init, U3_init]: initialization for the factor matrices U1 and U3 
    """
    if exacterrorbound is None:
        exacterrorbound = True
    if precision is None:
        precision = 1e-8
    if verbose is None:
        verbose = False
        
    SzX = X.shape
    # Set up and error checking on initial guess for U.
    if isinstance(init, list):
        Uinit = init
        if len(Uinit) != 2:
            raise ValueError('init does not have %d cells' % 2)
        if Uinit[0] is not None and Uinit[0].shape[0] != SzX[0]:
            raise ValueError('init{%d} is the wrong size' % 1)
        if Uinit[1] is not None and Uinit[1].shape[0] != SzX[2]:
            raise ValueError('init{%d} is the wrong size' % 2)
    else:
        Uinit = [None, None]
        if init == 'random':
            Uinit[1] = np.random.rand(SzX[2])
        elif init in ['nvecs', 'eigs']:
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (1 <= n <= N).
            print('  Computing leading e-vectors for factor %d.' % 2)
            Uinit[1] = tl.tenalg.svd_interface(tl.base.unfold(X,mode=2), n_eigenvecs=SzX[2])[0]
        else:
            raise ValueError('The selected initialization method is not supported')
    # Set up for iterations - initializing U and the fit.
    U1 = Uinit[0] if Uinit[0] is not None else np.eye(SzX[0])
    U3 = Uinit[1] if Uinit[1] is not None else np.eye(SzX[2])

    if verbose:
        print('\nTucker-2 for denoising:\n')
        
    normX2 = norm(X.flatten())**2
    
    if noiselevel >= 0:
        # Minimize rank or number of parameters with error bound constraints
        # min #param(Xhat) s.t. |X - [[G;U1 U3]] |_F^2 <= noiselevel^2 * numel(X)
        approx_bound = noiselevel**2 * np.prod(SzX)
        accuracy = normX2 - approx_bound
        rank_minimization = True
    else:
        # Minimize approximation error
        # min |X - [[G;U1 U3]] |_F^2
        rank_minimization = False
        accuracy = normX2
        
    # Permute
    permuteX = SzX[2] > SzX[0] * 1.5
    if permuteX:
        SzX = (SzX[2], SzX[1], SzX[0])
        X = np.transpose(X, (2, 1, 0))
        U1, U3 = U3, U1

    # print(approx_bound)
    
    # Reshape X
    X1 = X.reshape((X.shape[0], -1))
    X3 = X.reshape((-1, X.shape[2]))
    
    # Get the sizes of U1 and U3
    R = [U1.shape[1], U3.shape[1]]
    
    # Compute T and U2
    print(f'Ranks {R}')
    
    # Get the 2nd core tensor U2, given the first and third core tensors 
    T = U1.conj().T @ X1 # fix for complex valued tensor
    T = T.reshape((-1, X.shape[2]))
    U2 = T @ U3.conj() # fix for complex valued tensor
    U2 = U2.reshape((R[0], -1, R[1]))
    normU2_ = norm(U2.reshape(-1))**2
    
    # The loop updates U1 and U3, while U2 is computed after the iteration
    if rank_minimization:
        constrval = np.zeros(maxiters)
        rankR = np.zeros((maxiters, 2))
    objval = np.zeros(maxiters)
    
    for kiter in range(maxiters):
        # Update U1
        T = X3 @ U3.conj() # fix for complex valued tensor
        T = T.reshape((X.shape[0], -1))
        Q = T @ T.conj().T
        
        A = U1
        current_tracevalue = np.linalg.trace(A.conj().T @ Q @ A).real
        # print(f' Current trace {current_tracevalue} , Bound {accuracy}')

        u1, r1, trace_val = minrank_boundedtrace(Q, accuracy, exacterrorbound, precision)
        
        if u1 is not None:
            U1 = u1
            R[0] = r1
            normU2_ = trace_val
        
        # Update U3
        T = U1.conj().T @ X1
        T = T.reshape((-1, X.shape[2]))
        Q = T.conj().T @ T
        
        u3, r3, trace_val = minrank_boundedtrace(Q, accuracy, exacterrorbound, precision)
        
        if u3 is not None:
            U3 = u3.conj()
            R[1] = r3
            normU2_ = trace_val
        
        # Stop if converged
        if rank_minimization:
            # for the rank minimization problem 
            constrval[kiter] = normU2_
            objval[kiter] = X.shape[0] * R[0] + X.shape[2] * R[1] + X.shape[1] * R[0] * R[1]
            rankR[kiter, :] = R
        else:
            objval[kiter] = normU2_
        
        if verbose:
            if rank_minimization:
                print('%d  | (R1,R2) (%d, %d) |  #params %d | approx %.4g  | approx.bound %.4g' % (kiter, R[0], R[1], objval[kiter],(normX2-constrval[kiter])/normX2,approx_bound/normX2))
            else:
                # minimization of approximation error 
                print('%d  | (R1,R2) (%d, %d) |  #params %d'  % (kiter, R[0], R[1], objval[kiter]))
        
        if kiter > 0:
            isconverge = (abs(objval[kiter] - objval[kiter-1]) <= tol)
            if rank_minimization:
                isconverge = isconverge and (abs(constrval[kiter] - constrval[kiter-1]) <= tol)
            if isconverge:
                break
    
    # Compute U2
    U2 = T @ U3.conj()
    U2 = U2.reshape((R[0], -1, R[1]))
    
    # Remove extra values
    objval = objval[:kiter+1]
    if rank_minimization:
        constrval = constrval[:kiter+1]  # ||X - Un Phi(uk)||_F^2
        rankR = rankR[:kiter+1, :]
        aprxerror = normX2 - constrval
        noparams = objval
    else:
        noparams = X.shape[0] * R[0] + X.shape[2] * R[1] + X.shape[1] * R[0] * R[1]
        aprxerror = normX2 - objval
    
    # Permute if needed
    if permuteX:
        U2 = np.transpose(U2, (2, 1, 0))
        U1, U3 = U3, U1

    return U1,U2,U3,aprxerror,noparams,rankR



def tucker2_truncatedhosvd_init_complex_tensor(Y,approx_bound):
    normY = norm(Y.reshape(-1))
    # min rank(G) s.t. |X - [[G;U1 U3]] |_F^2 < noiselevel^2 * numel(X)
    # apx_bound = 0.01*normY**2
    # noiselevel = np.sqrt(apx_bound/Y.size)
    
    szY = Y.shape
    U1x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=0), n_eigenvecs=szY[0])[0]
    U3x = tl.tenalg.svd_interface(tl.base.unfold(Y,mode=2), n_eigenvecs=szY[2])[0]
    U2 = tl.tenalg.multi_mode_dot(Y,[U1x, U3x],modes = [0, 2], transpose=True)
    
    # find the smallest truncated model such that |U2(1:R1,:1:R2)|_F^2 <= apx_bound
    U2p = np.sum(np.abs(U2)**2, axis=1)
    U2p = np.cumsum(np.cumsum(U2p, axis=0), axis=1)
    # U2p[U2p < (normY**2 - apx_bound)] = np.nan
    
    R1 = np.arange(1, szY[0] + 1)
    R3 = np.arange(1, szY[2] + 1)
    R1x, R3x = np.meshgrid(R3, R1)
    numpararam = szY[0] * R1x + szY[2] * R3x + szY[1] * R1x * R3x
    
    # Find the indices where U2p >= (normY**2 - apx_bound)
    indices = np.where(U2p >= (normY**2 - approx_bound))
    
    # Extract the corresponding elements from numpararam
    numpararam_filtered = numpararam[indices]
    
    # Find the indices of the minimum element in B_filtered
    imin_index = np.argmin(numpararam_filtered)
    
    # print(imin_index)
    # # Convert the flattened index to the original shape
    # best_noparams = np.unravel_index(imin_index, B.shape)
    
    Rx = [R1[indices[0][imin_index]],R3[indices[1][imin_index]]]
    
    # print(Rx)
    # numpararam[U2p >= (normY**2 - apx_bound)] = np.nan
    # best_noparams, irr = np.nanmin(numpararam), np.nanargmin(numpararam)
    # irr = np.unravel_index(irr, U2p.shape)
    
    # rank TK2
    # Rx = [R1[irr[0]], R3[irr[1]]]
    
    # U2 = U2[0:Rx[0], :, 0:Rx[1]]
    # U2 = ttm(tensor(Y), [U1[:, 0:Rx[0]], U3[:, 0:Rx[1]]], [1, 3], 't')
    
    # approximation error 
    err = normY**2 - U2p[Rx[0]-1, Rx[1]-1]
    
    U1s, U3s = U1x[:, 0:Rx[0]], U3x[:, 0:Rx[1]]
 
    return U1s, U3s, err, U1x, U3x
