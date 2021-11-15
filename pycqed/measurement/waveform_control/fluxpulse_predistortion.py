import numpy as np
import scipy.signal as signal
import logging
import os
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
import logging
log = logging.getLogger(__name__)

def import_iir(filename):
    """
    imports csv files generated with Mathematica notebooks of the form
    a1_0,b0_0,b1_0
    a1_1,b0_1,b1_1
    a1_2,b0_2,b1_2
    .
    .
    .

    args:
        filename : string containging to full path of the file (or only the filename if in same directory)

    returns:
        [aIIRfilterLis,bIIRfilterList] : list of two numpy arrays compatable for use
        with the scipy.signal.lfilter() function
        used by filterIIR() function

    """
    IIRfilterList = np.loadtxt(filename,
                               delimiter=',')

    if len(IIRfilterList.shape) == 1:
        IIRfilterList = np.reshape(IIRfilterList,(1,len(IIRfilterList)))

    aIIRfilterList = np.transpose(np.vstack((np.ones(len(IIRfilterList)),
                                             -IIRfilterList[:,0])))
    bIIRfilterList = IIRfilterList[:,1:]

    return [aIIRfilterList,bIIRfilterList]


def filter_fir(kernel,x):
    """
    function to apply a FIR filter to a dataset

    args:
        kernel: FIR filter kernel
        x:      data set
    return:
        y :     data convoluted with kernel, aligned such that pulses do not
                shift (expects kernel to have a impulse like peak)
    """
    iMax = kernel.argmax()
    y = np.convolve(x,kernel,mode='full')[iMax:(len(x)+iMax)]
    return y


def filter_iir(aIIRfilterList, bIIRfilterList, x):
    """
    applies IIR filter to the data x (aIIRfilterList and bIIRfilterList are load by the importIIR() function)

    args:
        aIIRfilterList : array containing the a coefficients of the IIR filters
                         (one row per IIR filter with coefficients 1,-a1,-a2,.. in the form required by scipy.signal.lfilter)
        bIIRfilterList : array containing the b coefficients of the IIR filters
                         (one row per IIR filter with coefficients b0, b1, b2,.. in the form required by scipy.signal.lfilter)
        x : data array to be filtered

    returns:
        y : filtered data array
    """
    if len(aIIRfilterList[0]) == 3:  # second-order sections:
        sos = [list(b) + list(a) for a, b in zip(aIIRfilterList,
                                                 bIIRfilterList)]
        y = signal.sosfilt(sos, x)
    else:
        y = x
        for a, b in zip(aIIRfilterList, bIIRfilterList):
            y = signal.lfilter(b, a, y)
    return y


def gaussian_filter_kernel(sigma,nr_sigma,dt):
    """
    function to generate a Gaussian filter kernel with specified sigma and
    filter kernel width (nr_sigma).

    Args:
        sigma (float): width of the Gaussian
        nr_sigma (int): specifies the length of the filter kernel
        dt (float): AWG sampling period

    Returns:
        kernel (numpy array): Gaussian filter kernel
    """
    nr_samples = int(nr_sigma*sigma/dt)
    if nr_samples == 0:
        logging.warning('sigma too small (much smaller than sampling rate).')
        return np.array([1])
    gauss_kernel = signal.gaussian(nr_samples, sigma/dt, sym=False)
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)
    return np.array(gauss_kernel)


def scale_and_negate_IIR(filter_coeffs, scale):
    for i in range(len(filter_coeffs[0])):
        filter_coeffs[0][i][1] *= -1
    filter_coeffs[1][0] /= scale


def combine_FIR_filters(kernels, FIR_n_force_zero_coeffs=None):
    """
    Combine multiple FIR filter kernels to a single FIR filter kernel via
    convolution (optionally forcing some initial coefficients to zero).

    :param kernels: (list of arrays) the FIR filter kernels
    :param FIR_n_force_zero_coeffs: (int or None) If this is an int n,
        the first n coefficients of the combined filter are replaced by
        zeros and the filter is renormalized afterwards. This can, e.g., be
        used to create a causal filter.
    """
    if hasattr(kernels[0], '__iter__'):
        kernel_combined = kernels[0]
        for kernel in kernels[1:]:
            kernel_combined = np.convolve(kernel, kernel_combined)
        kernels = kernel_combined
    elif FIR_n_force_zero_coeffs is not None:
        kernels = deepcopy(kernels)  # make sure that we do not modify user input
    if FIR_n_force_zero_coeffs is not None:
        kernels[:FIR_n_force_zero_coeffs] = 0
        kernels /= np.sum(kernels)  # re-normalize
    return kernels


def convert_expmod_to_IIR(expmod, dt, inverse_IIR=True, direct=False):
    """
    Convert an exponential model A + B * exp(- t/tau) (or a list of such
    models) to a first-order IIR filter (or a list of such filters).

    :param expmod: list of exponential models  in the form
        [[A_0, B_0, tau_0], ... ], or a single exponential model
        [A_0, B_0, tau_0].
    :param dt: (float) AWG sampling period
    :param inverse_IIR: (bool, default: True) whether the IIR filters inverting
        the exponential model should be returned.
    :param direct: (bool, default: False) whether higher-order IIR filters
        should be implemented in directly. The default is to implement
        them as second-order sections.

    :return: A list of IIR filter coefficients in the form
        [aIIRfilterList, bIIRfilterList] according to the definition in
        filter_iir(). If expmod is a single first-order exponential model (or a
        single higher-order exponential model with direct=True), a single
        filter is returned in the form [a, b].
    """
    if hasattr(expmod[0], '__iter__'):
        iir = [convert_expmod_to_IIR(e, dt, inverse_IIR=inverse_IIR,
                                     direct=direct) for e in expmod]
        # Checking whether i[0][0] is iterable is needed because a single expmod
        # might be converted to multiple IIRs (second-order sections, SOS)
        a = np.concatenate(
            [[i[0]] if not hasattr(i[0][0], '__iter__') else i[0] for i in iir])
        b = np.concatenate(
            [[i[1]] if not hasattr(i[1][0], '__iter__') else i[1] for i in iir])
    else:
        A, B, tau = expmod
        if np.array(tau).ndim > 0:  # sum of exp mod
            import sympy as sp
            N = len(tau)
            a = [sp.Rational(a) for a in [A] + list(B)]
            tau_s = [sp.Rational(t / 1e-9) * 1e-9 for t in list(tau)]
            # In the next line, going via the reciprocal value is more precise
            # since we usually specify dt as 1 over sampling frequency.
            T = 1 / sp.Rational(f"{1 / dt}")
            if direct:
                z = sp.symbols('z')
                p = 2 / T * (1 - 1 / z) / (1 + 1 / z)
            else:
                p = sp.symbols('p')
                z = sp.exp(T * p)
            d = sp.prod([tau * p + 1 for tau in tau_s])
            r = [sp.prod([tau * p + 1 for i, tau in enumerate(tau_s)
                          if i != j]) for j in range(N)]
            n = (a[0] * d + sum([r * tau * a * p for r, tau, a
                                 in zip(r, tau_s, a[1:])]))
            if direct:
                n, d = sp.fraction((n / d).simplify(rational=True), exact=True)
                coeffs_n = n.as_poly(z).all_coeffs()
                coeffs_d = d.as_poly(z).all_coeffs()
                a = np.cast['float']([v.evalf() for v in coeffs_n])
                b = np.cast['float']([v.evalf() for v in coeffs_d])
                # further processing after the end of the if statement
            else:
                roots_n = n.as_poly(p).all_roots()
                # TODO: it seems that zeros can be complex even in the
                #  overdamped case. Double-check this!
                z_zeros = np.cast['complex128'](
                    [complex(z.subs(p, r).evalf()) for r in roots_n])
                z_poles = np.exp(dt * (- 1 / np.array(tau)))
                gain = sum([A] + list(B))
                if inverse_IIR:
                    sos = signal.zpk2sos(z_poles, z_zeros, 1 / gain)
                else:
                    sos = signal.zpk2sos(z_zeros, z_poles, gain)
                b = [sec[:3] for sec in sos]
                a = [sec[3:] for sec in sos]
                return [a, b]
        else:
            if 1 / tau < 1e-14:
                a, b = np.array([1, -1]), np.array([A + B, -(A + B)])
            else:
                a = np.array(
                    [(A + (A + B) * tau * 2 / dt), (A - (A + B) * tau * 2 / dt)])
                b = np.array([1 + tau * 2 / dt, 1 - tau * 2 / dt])
        if not inverse_IIR:
            a, b = b, a
        b = b / a[0]
        a = a / a[0]
    return [a, b]


def convert_IIR_to_expmod(filter_coeffs, dt, inverse_IIR=True):
    """
    Convert a first-order IIR filter (or a list of such filters) to an
    exponential model A + B * exp(- t/tau) (or a list of such models).

    :param filter_coeffs: IIR coefficients in the form
        [aIIRfilterList, bIIRfilterList] according to the definition in
        filter_iir(). Instead of a list, also single filter is accepted, i.e.,
        [a, b] will be interpreted as [[a], [b]].
    :param dt: (float) AWG sampling period
    :param inverse_IIR: (bool, default: True) whether the IIR filters should
        be interpreted as the filters inverting the exponential model

    :return: A list of exponential models is returned in the form
        [[A_0, B_0, tau_0], ... ], or a single exponential model
        [A_0, B_0, tau_0] if filter_coeffs are of the form [a, b].
    """
    if hasattr(filter_coeffs[0][0], '__iter__'):
        expmod = [convert_IIR_to_expmod([a, b], dt, inverse_IIR) for [a, b] in
                  zip(filter_coeffs[0], filter_coeffs[1])]
    else:
        [a, b] = filter_coeffs
        if not inverse_IIR:
            a, b = b, a
            b = b / a[0]
            a = a / a[0]
        gamma = np.mean(b)
        if np.abs(gamma) < 1e-14:
            A, B, tau =  1, 0, np.inf
        else:
            a_, b_ = a / gamma, b / gamma
            A = 1 / 2 * (a_[0] + a_[1])
            tau = 1 / 2 * (b_[0] - b_[1]) * dt / 2
            B = 1 / 2 * (a_[0] - a_[1]) * dt / (2 * tau) - A
        expmod = [A, B, tau]
    return expmod


def process_filter_coeffs_dict(flux_distortion, datadir=None, default_dt=None):
    """
    Prepares a distortion dictionary that can be stored into pulsar
    {AWG}_{channel}_distortion_dict based on information provided in a
    dictionary.

    :param flux_distortion: (dict) A dictionary of the format defined in
        QuDev_transmon.DEFAULT_FLUX_DISTORTION. In particular, the following
        keys FIR_filter_list and IIR_filter are processed by this function.
        They are list of dicts with a key 'type' and further keys.
        type 'csv': load filter from the file specified under the key
            'filename'. In case of an IIR filter, the filter will in
            addition be scaled by the value in the key 'scale_IIR'.
        type 'Gaussian': can be used for FIR filters. Gaussian kernel with
            parameters 'sigma', 'nr_sigma', and 'dt' specified in the
            respective keys. See gaussian_filter_kernel()
        type 'expmod': can be used for IIR filters. A filter that inverts an
            exponential model specified by the keys 'A', 'B', 'tau', and 'dt'.
            See convert_expmod_to_IIR().
        If flux_distortion in addition contains the keys FIR and/or IIR,
        the filters specified as described above will be appended to those
        already existing in FIR/IIR.
        If flux_distortion is already in the format to be stored in pulsar,
        it is returned unchanged.
        If flux_distortion contains the key FIR_n_force_zero_coeffs, its value
        is passed to combine_FIR_filters when combining FIR filters.
    :param datadir: (str) base dir for loading csv files. If None,
        it is assumed that the specified filename includes the full path.
    :param default_dt: (float) AWG sampling period to be used in cases where
        'dt' is needed, but not specified in a filter dict.

    """

    filterCoeffs = {}
    for fclass in 'IIR', 'FIR':
        filterCoeffs[fclass] = flux_distortion.get(fclass, [])
        if fclass == 'IIR' and len(filterCoeffs[fclass]) > 1:
            # convert coefficient lists into list of filters so that we can
            # append if needed
            filterCoeffs[fclass] = [[[a], [b]] for a, b in zip(
                filterCoeffs['IIR'][0], filterCoeffs['IIR'][1])]
        for f in flux_distortion.get(f'{fclass}_filter_list', []):
            if f['type'] == 'Gaussian' and fclass == 'FIR':
                coeffs = gaussian_filter_kernel(f.get('sigma', 1e-9),
                                                f.get('nr_sigma', 40),
                                                f.get('dt', default_dt))
            elif f['type'] == 'expmod' and fclass == 'IIR':
                expmod = f.get('expmod', None)
                if expmod is None:
                    expmod = [f.get('A'), f.get('B'), f.get('tau')]
                if not hasattr(expmod[0], '__iter__'):
                    expmod = [expmod]
                coeffs = convert_expmod_to_IIR(expmod,
                                               dt=f.get('dt', default_dt),
                                               direct=f.get('direct', False))
            elif f['type'] == 'csv':
                if datadir is not None:
                    filename = os.path.join(datadir,
                                            f['filename'].lstrip('\\'))
                else:
                    filename = f['filename']
                if (not os.path.exists(filename)
                        and a_tools.fetch_data_dir is not None
                        and filename.startswith(a_tools.datadir)):
                    # If the missing file is supposed to be stored inside
                    # the data folder, we can try to fetch it if a
                    # fetch_data_dir is configured in a_tools.
                    ts = filename.lstrip(a_tools.datadir).lstrip('\\')[
                         :15].replace('\\', '_')  # extract timestamp
                    a_tools.get_folder(ts)  # try to fetch the folder
                if fclass == 'IIR':
                    coeffs = import_iir(filename)
                    scale_and_negate_IIR(
                        coeffs,
                        f.get('scale_IIR', flux_distortion['scale_IIR']))
                else:
                    coeffs = np.loadtxt(filename)
            else:
                raise NotImplementedError(f"Unknown filter type {f['type']}")
            filterCoeffs[fclass].append(coeffs)

    if len(filterCoeffs['FIR']) > 0:
        filterCoeffs['FIR'] = [combine_FIR_filters(
            filterCoeffs['FIR'],
            FIR_n_force_zero_coeffs=flux_distortion.get(
                'FIR_n_force_zero_coeffs', None))]
    else:
        del filterCoeffs['FIR']
    if len(filterCoeffs['IIR']) > 0:
        filterCoeffs['IIR'] = [
            np.concatenate([i[0] for i in filterCoeffs['IIR']]),
            np.concatenate([i[1] for i in filterCoeffs['IIR']])]
    else:
        del filterCoeffs['IIR']
    return filterCoeffs
