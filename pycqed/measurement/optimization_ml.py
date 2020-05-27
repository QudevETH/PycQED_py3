import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from pycqed.analysis import machine_learning_toolbox as ml
except Exception:
    logger.warning('Machine learning packages not loaded. '
                   'Run from pycqed.analysis import machine_learning_toolbox to see errors.')

from scipy.optimize import fmin_l_bfgs_b,fmin,minimize,fsolve

from pycqed.measurement import optimize


def neural_network_opt(fun, training_grid, target_values = None,
                       estimator='GRNN_neupy',hyper_parameter_dict=None,
                       x_init = None):
    """
    parameters:
        fun:           Function that can be used to get data points if None,
                       target_values have to be provided instead.
        training_grid: The values on which to train the Neural Network. It
                       contains features as column vectors of length as the
                       number of datapoints in the training set.
        target_values: The target values measured during data acquisition by a
                       hard sweep over the traning grid.
        estimator: The estimator used to model the function mapping the
                   training_grid on the target_values.
        hyper_parameter_dict: if None, the default hyperparameters
                              of the selected estimator are used. Should contain
                              estimator dependent hyperparameters such as hidden
                              layer sizes for a neural network. See
                              <machine_learning_toolbox> for specific
                              information on available estimators.
        x_ini: Initial values for the minimization of the fitted function.
    output:
        optimal points where network is minimized.
        est: estimator instance representing the trained model. Consists of a
             predict(X) method, which computes the network response for a given
             input value X.
    """
    ###############################################################
    ###          create measurement data from test_grid         ###
    ###############################################################
    #get input dimension, training grid contains parameters as row(!!) vectors
    if len(np.shape(training_grid)) == 1:
        training_grid = np.transpose(np.array([training_grid]))
    n_samples = np.size(training_grid,0)
    print('Nr Samples: ', n_samples)
    n_features = np.size(training_grid,1)
    print('Nr Features: ', n_features)

    if fun is None:
        output_dim = np.size(target_values,1)
    else:
        #if the sweep is adaptive, acquire data points by applying fun
        first_value = fun(training_grid[0])
        output_dim = np.size(first_value)
        target_values = np.zeros((n_samples,output_dim))
        target_values[0,:] = first_value
        for i in range(1,n_samples):
            target_values[i,:]=fun(training_grid[i])

    #Preprocessing of Data. Mainly transform the data to mean 0 and interval [-1,1]
    training_grid_centered,target_values_centered,\
    input_feature_means,input_feature_ext,\
    output_feature_means,output_feature_ext \
                 = center_and_scale(training_grid,target_values)
    #Save the preprocessing information in order to be able to rescale the values later.
    pre_processing_dict ={'output': {'scaling': output_feature_ext,
                                     'centering':output_feature_means},
                          'input': {'scaling': input_feature_ext,
                                    'centering':input_feature_means}}

    ##################################################################
    ### initialize grid search cross val with hyperparameter dict. ###
    ###    and MLPR instance and fit a model functione to fun()     ###
    ##################################################################
    def mlpr():
        est = ml.MLP_Regressor_scikit(hyper_parameter_dict,
                                      output_dim=output_dim,
                                      n_feature=n_samples,
                                      pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered, np.ravel(target_values_centered))
        est.print_best_params()
        return est

    def dnnr():
        est = ml.DNN_Regressor_tf(hyper_parameter_dict,
                               output_dim=output_dim,
                               n_feature=n_features,
                               pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered,target_values_centered)
        return est

    def grnn():
        est = ml.GRNN_neupy(hyper_parameter_dict,
                            pre_proc_dict=pre_processing_dict)
        cv_est = ml.CrossValidationEstimator(hyper_parameter_dict,est)
        cv_est.fit(training_grid_centered,target_values_centered)
        return cv_est

    def polyreg():
        est = ml.Polynomial_Regression(hyper_parameter_dict,
                                       pre_proc_dict=pre_processing_dict)
        est.fit(training_grid_centered,target_values_centered)
        return est

    estimators = {'MLP_Regressor_scikit': mlpr, #defines all current estimators currently implemented
                  'DNN_Regressor_tf': dnnr,
                  'GRNN_neupy': grnn,
                  'Polynomial_Regression_scikit': polyreg}

    est = estimators[estimator]()       #create and fit instance of the chosen estimator

    def estimator_wrapper(X):
        pred = est.predict([X])
        print('pred: ', pred)
        if output_dim == 1.:
            return np.abs(pred+1.)
        else:
            pred = pred[0]
            norm = 0.
            for it in range(len(pred)):
                print(it)
                if it == 0:
                    w = 1
                else:
                    w = 1
                norm += w*np.abs(pred[it] + 1.)
            output = norm
            print('norm: ', norm)
            print('')

            return output

    ###################################################################
    ###     perform gradient descent to minimize modeled landscape  ###
    ###################################################################
    if x_init is None:
        x_init = np.zeros(n_features)
        #The data is centered. No values above -1,1 should be encountered
        bounds=[(-1.,1.) for i in range(n_features)]
        res = fmin_l_bfgs_b(estimator_wrapper, x_init, bounds=bounds,
                            approx_grad=True)
    else:
        print('x_init minimizer:', x_init)
        for it in range(n_features):
            x_init[it] = (x_init[it]-input_feature_means[it])/input_feature_ext[it] # scale initial value
        bounds=[(-1.,0.5) for i in range(n_features)]
        res = fmin_l_bfgs_b(estimator_wrapper, x_init, approx_grad=True, bounds=bounds)
        # res = minimize(estimator_wrapper, x_init, method='Nelder-Mead')
        # res = [res.x]
        print('result:', res)
        # print(res)


    # result = res.x
    result = res[0]
    opti_flag = True

    for it in range(n_features):
        if not opti_flag:
            break
        if np.abs(result[it]) >= 2*np.std(training_grid_centered, 0)[it]:
            opti_flag = False
    if not opti_flag:
        print('optimization most likely failed. Results outside 2-sigma surrounding'
              'of at least one data feature mean value! Values will still be updated.')
    # Rescale values
    amp = est.predict([result])[0]
    print('amp: ', amp)
    if output_dim == 1.:
        amp = amp*output_feature_ext+output_feature_means
    else:
        for it in range(output_dim):
            amp[it] = amp[it]*output_feature_ext[it]+output_feature_ext[it]
    for it in range(n_features):
        result[it] = result[it]*input_feature_ext[it]+input_feature_means[it]
    print('minimization results: ', result, ' :: ', amp)

    return np.array(result), est,opti_flag