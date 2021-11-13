import logging
log = logging.getLogger(__name__)
from collections import OrderedDict
from copy import copy, deepcopy
import numpy as np
from numpy import array  # Needed for eval. Do not remove.

class SweepPoints(list):
    """
    This class is used to create sweep points for any measurement.
    The SweepPoints object is a list of dictionaries of the form:
        [
            # 1st sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},

            # 2nd sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},

             .
             .
             .

            # D-th sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},
        ]

    Example how to use this class to create a 2D sweep for 3 qubits, where
    the first (hard) sweep is over amplitudes and the :

    sp = SweepPoints()
    for qb in ['qb1', 'qb2', 'qb3']:
        sp.add_sweep_parameter(f'lengths_{qb}', np.linspace(10e-9, 1e-6, 80),
        's', 'Pulse length, $L$')
    sp.add_sweep_dimension()
    for qb in ['qb1', 'qb2', 'qb3']:
        sp.add_sweep_parameter(f'amps_{qb}', np.linspace(0, 1, 20),
        'V', 'Pulse amplitude, $A$')
    """
    def __init__(self, param=None, values=None, unit='', label=None,
                 dimension=-1, min_length=0):
        """
        Create a SweepPoints instance.
        :param param: list of dicts, repr of SweepPoints instance, or name of
            a sweep parameter
        :param values: list or array of numeric values of the sweep parameter
            specified by param
        :param unit: string specifying the unit of the sweep parameter
            specified by param
        :param label: str specifying the (latex style) label/name of the sweep
            parameter specified by param
        :param dimension: sweep dimension where sweep parameter param should be
            added
        :param min_length: minimum number of sweep dimensions to create
        """
        super().__init__()
        if isinstance(param, list):
            self.add_dict_list(param)
        elif isinstance(param, str):
            if values is not None:
                self.add_sweep_parameter(param, values, unit, label,
                                         dimension)
            else:
                self.add_dict_list(eval(param))

        while len(self) < min_length:
            self.add_sweep_dimension()

    def __getitem__(self, i):
        """
        Overloading of List.__getitem__ to ensure type SweepPoints is preserved.
        :param i: element or slice
        :return: element or new SweepPoints instance
        """
        if isinstance(i, str):
            new_data = self.get_sweep_params_property('values', 'all', i)
        else:
            new_data = super().__getitem__(i)
            if isinstance(i, slice):
                new_data = self.__class__(new_data)
        return new_data

    def add_dict_list(self, dict_list):
        """
        Append the dicts in dict_list to self.
        :param dict_list: list of dictionaries in the format of this class, or
            in the legacy format {param_name: {'values': ...,
                                               'unit': ...,
                                               'label': ...}}
        """
        for d in deepcopy(dict_list):
            if len(d) == 0 or isinstance(list(d.values())[0], tuple):
                # assume that dicts have the same format as this class
                self.append(d)
            else:
                # import from a list of sweep dicts in the old format
                self.append({k: (v['values'],
                                 v.get('unit', ''),
                                 v.get('label', k))
                             for k, v in d.items()})

    def add_sweep_parameter(self, param_name, values, unit='', label=None,
                            dimension=-1):
        """
        Adds sweep points to a given dimension.
        :param param_name: (str) parameter name
        :param values: (list or numpy array) sweep values
        :param unit: (optional str) unit of the values (default: '')
        :param label: (optional str) label e.g. for plots (default: param_name)
        :dim: the dimension to which the point should be added (default:
            last dimension)
        """
        if label is None:
            label = param_name
        assert self.find_parameter(param_name) is None, \
            f'A sweep parameter with name "{param_name}" already exists.'
        while len(self) == 0 or (dimension >= 0 and dimension >= len(self)):
            self.add_sweep_dimension()
        assert self.length(dimension) in [0, len(values)], \
            'Number of values has to match the length of existing sweep ' \
            'points.'
        self[dimension].update({param_name: (values, unit, label)})

    def add_sweep_dimension(self):
        self.append(dict())

    def get_sweep_dimension(self, dimension='all', pop=False):
        """
        Returns the sweep dict of the sweep dimension specified by dimension.
        :param dimension: int specifying a sweep dimension or
            the string 'all'
        :param pop: bool specifying whether to pop (True) or get(False) the
            sweep dimension.
        :return: self if dimension == 'all', else self[dimension]
        """
        if dimension == 'all':
            if pop:
                to_return = deepcopy(self)
                self.clear()
                return to_return
            else:
                return self
        else:
            if len(self) < dimension:
                raise ValueError(f'Dimension {dimension} not found.')
            to_return = self[dimension]
            if pop:
                self[dimension] = {}
            return to_return

    def get_sweep_params_description(self, param_names, dimension='all',
                                     pop=False):
        """
        Get the sweep tuples for the sweep parameters param_names if they are
        found in the sweep dimension dict specified by dimension.
        :param param_names: string or list of strings corresponding to keys in
            the dictionaries in self. Can also be 'all'
        :param dimension: 'all' or int specifying a sweep dimension
        :param pop: bool specifying whether to pop (True) or get(False) the
            sweep parameters.
        :return:
            If the param_names are found in self or self[dimension]:
            if param_names == 'all': list with all the sweep tuples
                in the sweep dimension dict specified by dimension.
            if param_names is string: string with the sweep tuples of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is list: list with the sweep tuples of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is None: string corresponding to the
                first sweep parameter in the sweep dimension dict
            If none of param_names are found, raises KeyError.
        """
        sweep_points_dim = self.get_sweep_dimension(
            dimension, pop=pop and param_names == 'all')
        is_list = True
        if param_names != 'all' and not isinstance(param_names, list):
            param_names = [param_names]
            is_list = False

        sweep_param_values = []
        if isinstance(sweep_points_dim, list):
            for sweep_dim_dict in sweep_points_dim:
                if param_names == 'all':
                    sweep_param_values += list(sweep_dim_dict.values())
                else:
                    for pn in param_names:
                        if pn in sweep_dim_dict:
                            sweep_param_values += [sweep_dim_dict.pop(pn) if pop
                                                   else sweep_dim_dict[pn]]
        else:  # it is a dict
            if param_names == 'all':
                sweep_param_values += list(sweep_points_dim.values())
            else:
                for pn in param_names:
                    if pn in sweep_points_dim:
                        sweep_param_values += [sweep_points_dim.pop(pn) if pop
                                               else sweep_points_dim[pn]]
        if len(sweep_param_values) == 0:
            s = "sweep points" if dimension == "all" else f'sweep dimension ' \
                                                          f'{dimension}'
            raise KeyError(f'{param_names} not found in {s}.')

        if is_list:
            return sweep_param_values
        else:
            return sweep_param_values[0]

    def get_sweep_params_property(self, property, dimension='all',
                                  param_names=None):
        """
        Get a property of the sweep parameters param_names in self.
        :param property: str with the name of a sweep param property. Can be
            "values", "unit", "label."
        :param dimension: 'all' or int specifying a sweep dimension
            (default 'all')
        :param param_names: None, or string or list of strings corresponding to
            keys in the sweep dimension specified by dimension.
            Can also be 'all'
        :return:
            if param_names == 'all': list with the property of all
                param_names in the sweep dimension dict specified by dimension.
            if param_names is string: the property of the sweep parameter
                specified in param_names in the sweep dimension dict specified
                by dimension.
            if param_names is list: list with the property of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is None: property corresponding to the
                first sweep parameter in the sweep dimension dict
        """
        properties_dict = {'values': 0, 'unit': 1, 'label': 2}
        sweep_points_dim = self.get_sweep_dimension(dimension)

        if param_names is None:
            if isinstance(sweep_points_dim, list):
                for sweep_dim_dict in sweep_points_dim:
                    if len(sweep_dim_dict) == 0:
                        return [] if property == 'values' else ''
                    else:
                        return next(iter(sweep_dim_dict.values()))[
                            properties_dict[property]]
            else:
                if len(sweep_points_dim) == 0:
                    return [] if property == 'values' else ''
                else:
                    return next(iter(sweep_points_dim.values()))[
                        properties_dict[property]]
        elif param_names == 'all' or isinstance(param_names, list):
            return [pnd[properties_dict[property]] for pnd in
                    self.get_sweep_params_description(param_names,
                                                      dimension)]
        else:
            return self.get_sweep_params_description(
                param_names, dimension)[properties_dict[property]]

    def get_meas_obj_sweep_points_map(self, measurement_objects):
        """
        Constructs the measurement-objects-sweep-points map as the dict
        {mobj_name: [sweep_param_name_0, ..., sweep_param_name_n]}

        If a sweep dimension has only one sweep parameter name (dict with only
        one key), then it assumes all mobjs use that sweep parameter name.

        If the sweep dimension has more than one sweep parameter name (dict with
        several keys), then:
            - first adds to the list for each mobj only those sweep
            param names that contain the mobj_name.

            ! Currently assumes the mobj names substrings are separated by '_'
            from each other and from the rest of the substrings in
            the sweep parameter names. So for example, qb1qb9_amplitude will not
            be found for either qb1 or qb9. Neither will qb8 be found in
            qb8amplitude !

            - if some parameters remain that do not contain any of the
            measurement_objects names, it is assumed that all
            measurement_objects used them so they will be added for each mobj

        :param measurement_objects: list of strings to be used as keys in the
            returned dictionary. These are the measured object names.
            Can also be list of measurement object instances with a name
            attribute, in which case this function gets the list of names.
        :return: dict of the form
         {mobj_name: [sweep_param_name_0, ..., sweep_param_name_n]}
        """

        # Ensure measurement_objects is a list
        if isinstance(measurement_objects, list):
            measurement_objects = copy(measurement_objects)
        else:
            measurement_objects = [measurement_objects]

        for i, mobj in enumerate(measurement_objects):
            if hasattr(mobj, 'name'):
                # A list of measurement object instances with a
                # name attribute was provided
                measurement_objects[i] = mobj.name

        sweep_points_map = {mobjn: [] for mobjn in measurement_objects}

        for dim, d in enumerate(self):
            # d is a dictionary with keys == sweep parameter names
            if len(d) == 1:
                # Only one sweep parameter.
                # Assume all mobjs use the same param_name given by the key
                # of d.
                for mobjn in measurement_objects:
                    sweep_points_map[mobjn] += [next(iter(d))]
            else:
                all_pars = []
                for i, mobjn in enumerate(measurement_objects):
                    # Find all sweep param names that contain the mobj name.
                    # Assumes the mobj names substrings are separated by '_'
                    # from each other and from the rest of the substrings in
                    # the sweep parameter name
                    pars = [k for k in list(d) if mobjn in k.split('_')]
                    if len(pars):
                        # Append found sweep param names to the sweep_points_map
                        # for this mobj
                        sweep_points_map[mobjn] += pars
                        
                    # Collect all found pars, to be used below
                    all_pars += pars

                # Find the remaining sweep parameter names in this dimension
                # that do not contain any of the mobj names. These are assumed
                # to be used by all measurement_objects and will be appended
                # in sweep_points_map for each mobj.
                remaining_pars = [k for k in list(d) if k not in all_pars]
                if len(remaining_pars):
                    for mobjn in measurement_objects:
                        sweep_points_map[mobjn] += remaining_pars

        return sweep_points_map

    def length(self, dimension='all'):
        """
        Returns the number of sweep points in a given sweep dimension (after a
        sanity checking).

        :param dimension: ('all' or int) sweep dimension (default: 'all').

        :return: (int) number of sweep points in the given dimension
        """

        if dimension == 'all':
            return [self.length(d) for d in range(len(self))]

        if len(self) == 0 or (dimension >= 0 and dimension >= len(self)):
            return 0
        n = 0
        for p in self[dimension].values():
            if n == 0:
                n = len(p[0])
            elif n != len(p[0]):
                raise ValueError('The lengths of the sweep points are not '
                                 'consistent.')
        return n

    def update(self, sweep_points):
        """
        Updates the sweep dictionaries of all dimensions with the sweep
        dictionaries passed as sweep_points. Non-existing
        parameters and required additional dimensions are added if needed.

        :param sweep_points: (SweepPoints) a SweepPoints object containing
            the sweep points to be updated.

        :return:
        """
        while len(self) < len(sweep_points):
            self.add_sweep_dimension()
        for d, u in zip(self, sweep_points):
            d.update(u)

    def update_property(self, param_names, values=None,
                        units=None, labels=None):
        """
        Updates sweep properties (values, units, or labels) of the sweep
        parameters in param_names.

        :param param_names: list of sweep param names in self
        :param values: (list) contains arrays of values
            corresponding to param_names that will replace the existing values
        :param units: (list) units corresponding to param_names that will
            replace the existing units
        :param labels: (list) labels corresponding to param_names that will
            replace the existing labels

        Assumes order in values/units/labels corresponds to the order in
        param_names.

        :return:
        """

        if not len(param_names):
            # nothing to update
            return

        if values is None and units is None and labels is None:
            # nothing to update
            return

        dims = [self.find_parameter(par) for par in param_names]
        if values is None or not len(values):
            values = self.get_sweep_params_property(
                'values', param_names=param_names)
        if units is None or not len(values):
            units = self.get_sweep_params_property(
                'unit', param_names=param_names)
        if labels is None or not len(values):
            labels = self.get_sweep_params_property(
                'label', param_names=param_names)

        lengths = [len(p) for p in [param_names, values, units, labels]]
        if np.unique(lengths).size > 1:
            raise ValueError('param_names, values, units, labels have '
                             'must have the same number of entries.')

        # remove the sweep parameters whose properties will be updated
        for par in param_names:
            self.remove_sweep_parameter(par)

        # add the sweep parameters again with new properties]
        for i, par in enumerate(param_names):
            self.add_sweep_parameter(par, values[i], units[i],
                                     labels[i], dims[i])

    def find_parameter(self, param_name):
        """
        Returns the index of the first dimension in which a given sweep
        parameter is found.

        :param param_name: (str) name of the sweep parameter

        :return: (int or None) the index of the first dimension in which the
            parameter if found or None if no parameter with the given name
            exists.
        """
        for dim in range(len(self)):
            if param_name in self[dim]:
                return dim
        return None

    def subset(self, i, dimension=0):
        """
        Returns a new SweepPoints object with one of the dimensions reduced
        to a subset of the sweep values. The other dimensions are unchanged.
        :param i: (list) indices of the sweep values that should be
            contained in the subset.
        :param dimension: (int, default 0) index of the dimension that
            should be reduced
        """
        sp = SweepPoints(self)
        for k, v in sp[dimension].items():
            sp[dimension][k] = (np.array(v[0])[i], v[1], v[2])
        return sp

    def remove_sweep_parameter(self, param_name):
        """
        Removes a sweep parameter with a given name from the SweepPoints
        object. If the parameter is not found, a warning is issued.
        :param param_name: (str) name of the sweep parameter to remove
        """
        dim = self.find_parameter(param_name)
        if dim is None:
            log.warning(f"remove_sweep_parameter: Sweep parameter "
                        f"{param_name} not found.")
        else:
            del self[dim][param_name]
