import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)

from sklearn.model_selection import GridSearchCV as gcv, train_test_split

from scipy.optimize import fmin_l_bfgs_b,fmin,minimize,fsolve

def nelder_mead(fun, x0,
                initial_step=0.1,
                no_improve_thr=10e-6, no_improv_break=10,
                maxiter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                verbose=False):
    '''
    parameters:
        fun (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x0
        x0 (numpy array): initial position
        initial_step (float/np array): determines the stepsize to construct
            the initial simplex. If a float is specified it uses the same
            value for all parameters, if an array is specified it uses
            the specified step for each parameter.

        no_improv_thr,  no_improv_break (float, int): break after
            no_improv_break iterations with an improvement lower than
            no_improv_thr
        maxiter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        alpha (float): reflection coefficient
        gamma (float): expansion coefficient
        rho (float): contraction coefficient
        sigma (float): shrink coefficient
            For details on these parameters see Wikipedia page

    return: tuple (best parameter array, best score)


    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Implementation from https://github.com/fchollet/nelder-mead, edited by
    Adriaan Rol for use in PycQED.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    '''
    # init
    x0 = np.array(x0)  # ensures algorithm also accepts lists
    dim = len(x0)
    prev_best = fun(x0)
    no_improv = 0
    res = [[x0, prev_best]]
    if type(initial_step) is float:
        initial_step_matrix = np.eye(dim)*initial_step
    elif (type(initial_step) is list) or (type(initial_step) is np.ndarray):
        if len(initial_step) != dim:
            raise ValueError('initial_step array must be same lenght as x0')
        initial_step_matrix = np.diag(initial_step)
    else:
        raise TypeError('initial_step ({})must be list or np.array'.format(
                        type(initial_step)))

    for i in range(dim):
        x = copy.copy(x0)
        x = x + initial_step_matrix[i]
        score = fun(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after maxiter
        if maxiter and iters >= maxiter:
            # Conclude failure break the loop
            if verbose:
                print('max iterations exceeded, optimization failed')
            break
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            # Conclude success, break the loop
            if verbose:
                print('No improvement registered for {} rounds,'.format(
                      no_improv_break) + 'concluding succesful convergence')
            break

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = fun(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = fun(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = fun(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = fun(redx)
            nres.append([redx, score])
        res = nres

    # once the loop is broken evaluate the final value one more time as
    # verification
    fun(res[0][0])
    return res[0]


def SPSA(fun, x0,
         initial_step=0.1,
         no_improve_thr=10e-6, no_improv_break=10,
         maxiter=0,
         gamma=0.101, alpha=0.602, a=0.2, c=0.3, A=300,
         p=0.5, ctrl_min=0.,ctrl_max=np.pi,
         verbose=False):
    '''
    parameters:
        fun (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x0
        x0 (numpy array): initial position


        no_improv_thr,  no_improv_break (float, int): break after
            no_improv_break iterations with an improvement lower than
            no_improv_thr
        maxiter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        alpha, gamma, a, c, A, (float): parameters for the SPSA gains
            (see refs for definitions)
        p (float): probability to get 1 in Bernoulli +/- 1 distribution
            (see refs for context)
        ctrl_min, ctrl_max (float/array): boundaries for the parameters.
            can be either a global boundary for all dimensions, or a
            numpy array containing the boundary for each dimension.
    return: tuple (best parameter array, best score)

    alpha, gamma, a, c, A and p, are parameters for the algorithm.
    Their function is described in the references below,
    and even optimal values have been discussed in the literature.


    Pure Python/Numpy implementation of the SPSA algorithm designed by Spall.
    Implementation from http://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF,
    edited by Ramiro Sagastizabal for use in PycQED.
    Reference: http://www.jhuapl.edu/SPSA/Pages/References-Intro.htm
    '''
    # init
    x0 = np.array(x0)  # ensures algorithm also accepts lists
    dim = len(x0)
    prev_best = fun(x0)
    no_improv = 0
    res = [[x0, prev_best]]

    x = copy.copy(x0)

    # SPSA iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after maxiter
        if maxiter and iters >= maxiter:
            # Conclude failure break the loop
            if verbose:
                print('max iterations exceeded, optimization failed')
            break
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            # Conclude success, break the loop
            if verbose:
                print('No improvement registered for {} rounds,'.format(
                      no_improv_break) + 'concluding succesful convergence')
            break
        # step 1
        a_k = a/(iters+A)**alpha
        c_k = c/iters**gamma
        # step 2
        delta = np.where(np.random.rand(dim) > p, 1, -1)
        # step 3
        x_plus = x+c_k*delta
        x_minus = x-c_k*delta
        y_plus = fun(x_plus)
        y_minus = fun(x_minus)
        # res.append([x_plus, y_plus])
        # res.append([x_minus, y_minus])
        # step 4
        gradient = (y_plus-y_minus)/(2.*c_k*delta)
        # step 5
        x = x-a_k*gradient
        x = np.where(x < ctrl_min, ctrl_min, x)
        x = np.where(x > ctrl_max, ctrl_max, x)
        score = fun(x)
        res.append([x, score])

    # once the loop is broken evaluate the final value one more time as
    # verification
    fun(res[0][0])
    return res[0]

def generate_new_training_set(new_train_values, new_target_values,
                              training_grid=None, target_values=None):
    if training_grid is None:
        training_grid =new_train_values
        target_values = new_target_values

    else:
        if np.shape(new_train_values)[1] != np.shape(training_grid)[1] or \
           np.shape(new_target_values)[1] != np.shape(target_values)[1]:
            print('Shape missmatch between new training values and existing ones!'
                  ' Returning None.')
            return None,None

        training_grid = np.append(training_grid,new_train_values,axis=0)
        target_values = np.append(target_values,new_target_values,axis=0)

    return training_grid,target_values

def center_and_scale(X_in,y_in):
    '''
    Preprocessing of Data. Mainly transform the data to mean 0 and interval [-1,1]
    :param X: training data list of parameters (each equally long). Standing vector!
    :param y: validation data list of parameters (each equally long).Standing vector!
    :output:
        :X: rescaled and centered training data
        :y: rescaled and centered test data
        :input_feature_means: mean values of initial training data parameters
        :output_feature_means: mean values of initial validation data parameters
        :input_feature_ext: abs(max-min) of initial training data parameters
        :output_feature_ext: abs(max-min) of initial validation data parameters
    '''

    if not isinstance(X_in,np.ndarray):
        X_in = np.array(X_in)
    if X_in.ndim == 1:
        X_in.shape = (np.size(X_in),X_in.ndim)
        #X_in.reshape((np.size(X_in),X_in.ndim))
    if not isinstance(y_in,np.ndarray):
        y_in= np.array(y_in)
    if y_in.ndim == 1:
        #y_in.reshape((np.size(y_in),y_in.ndim))
        y_in.shape = (np.size(y_in),y_in.ndim)

    X = copy.deepcopy(X_in)
    y = copy.deepcopy(y_in)
    input_feature_means = np.zeros(np.size(X,1))       #saving means of training
    output_feature_means = np.zeros(np.size(y,1))     #and target features
    input_feature_ext= np.zeros(np.size(X,1))
    output_feature_ext = np.zeros(np.size(y,1))

    if  np.size(X,1)==1:
        input_feature_means= [np.mean(X)]
        input_feature_ext = [np.max(X) \
                             -np.min(X)]
        X -= input_feature_means  #offset to mean 0
        X /= input_feature_ext    #rescale to [-1,1]

    else:
        for it in range(np.size(X,1)):
            input_feature_means[it]= np.mean(X[:,it])
            input_feature_ext[it] = np.max(X[:,it]) \
                                    -np.min(X[:,it])
            X[:,it] -= input_feature_means[it]  #offset to mean 0
            X[:,it] /= input_feature_ext[it]    #rescale to [-1,1]
    if np.size(y,1) == 1:
        output_feature_means= [np.mean(y)]
        output_feature_ext = [np.max(y) \
                              -np.min(y)]
        y -= output_feature_means #offset to mean 0
        y /= output_feature_ext   #rescale to [-1,1]
    else:
        for it in range(np.size(y,1)):
            output_feature_means[it]= np.mean(y[:,it])
            output_feature_ext[it] = np.max(y[:,it]) \
                                     -np.min(y[:,it])
            y[:,it] -= output_feature_means[it] #offset to mean 0
            y[:,it] /= output_feature_ext[it]   #rescale to [-1,1]
    return X,y,\
           input_feature_means,input_feature_ext,\
           output_feature_means,output_feature_ext




def gradient(fun,x,grid_spacing):
    """
    this computes the gradient of fun() using finite differences
    :param fun: function of which to compute the gradient
    :param x: evaluation point for the gradient
    :param grid_spacing: displacement in forward finite difference, has to be
                         provided for every feature in x
    :return: returns gradient value grad(fun)(x) computed by finite differences
    """

    grad = np.zeros(len(x))
    for iter in range(len(x)):
        x_h = np.array(x)
        #using forward difference here
        x_h[iter] += grid_spacing[iter]
        #compute finite difference
        grad[iter]=(fun(x_h)-fun(x))/grid_spacing[iter]
    return grad


def gradient_descent(fun, x_ini,grid_spacing,lamb_ini=1, max_iter=500 ):
    """
    :param fun: function to be minimized
    :param x_ini: initial point for the iteration 
    :param grid_spacing: spacing between points on the discretized evaluation 
                         grid. has to provide a spacing for every feature in x_ini.
    :param lamb_ini: initial learning rate. 
    :param tol: tolerance to determine convergence. 
    :param max_iter: maximum iterations permitted until gradient descent fails
    
    :return: input values that minimizes fun()

    Note: using Barzilai-Borwein adaptive step lengths for second derivative approx.
    """
    iter = 0
    #determine the tolerance as 1e-4 times the norm of the step size vector
    tol = np.linalg.norm(grid_spacing)*1e-4
    #perform first step
    x = np.array(x_ini)
    grad = gradient(fun,x,grid_spacing)
    s = -grad/lamb_ini         #go downhill
    diff = np.linalg.norm(s)   #defines the stepsize taken. If < tol, extremum was found
    x_new = np.array(x+s)

    while diff > tol and iter < max_iter:
        #central loop, makes gradient update and performs step with adaptive
        #learning rate.
        grad_new = gradient(fun,x_new,grid_spacing)
        y = grad_new - grad
        grad = np.array(grad_new)
        lamb = np.linalg.norm(y)**2/np.dot(y,s)
        s = -grad_new/lamb
        x = np.array(x_new)
        x_new = np.array(x+s)
        diff = np.linalg.norm(s)
        iter += 1
        if(diff <= tol and iter < max_iter):
            print("gradient descent finished after "+str(iter)+"iterations")
            print("minimum found at: "+str(x_new))
        elif(iter>=max_iter):
            print("Warning: gradient descent did not finish within \
                       max iterations!")
    return x_new,[diff,iter]
