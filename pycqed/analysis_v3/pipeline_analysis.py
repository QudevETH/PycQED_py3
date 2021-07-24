import traceback
import logging
log = logging.getLogger(__name__)

# analysis_v3 node modules
from pycqed.analysis_v3 import saving as save_mod
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline

search_modules = set()
search_modules.add(hlp_mod)


def process_pipeline(data_dict, processing_pipeline=None, append_pipeline=False,
                     save_processed_data=True, save_figures=True, **params):
    """
    Calls all the classes/functions found in processing_pipeline,
    which is a list of dictionaries of the form:

    [
        {'node_name': function_name0, **node_params0},
        {'node_name': function_name1, **node_params1},
    ]

    All node functions must exist in the modules specified in the global vaiable
    "search_modules" define at the top of this module, and will process the
    data corresponding to the keys specified as "keys_in" in the **node_params
    of each node.

    Each node in the pipeline will put the processed data in the data_dict,
    under the key(s)/dictionary key path(s) specified in 'keys_out' in the
    the **node_params of each node.
    """

    # Add flag that this is an analysis_v3 data_dict. This is used by the
    # Saving class.
    if 'is_data_dict' not in data_dict:
        data_dict['is_data_dict'] = True

    if processing_pipeline is None:
        processing_pipeline = hlp_mod.get_param('processing_pipeline',
                                                data_dict, raise_error=True)
    elif append_pipeline:
        for node_params in processing_pipeline:
            hlp_mod.add_param('processing_pipeline', [node_params],
                              data_dict, append_value=True)

    # Instantiate a ProcessingPipeline instance in case it is an ordinary list
    processing_pipeline = ProcessingPipeline(processing_pipeline)

    for node_params in processing_pipeline:
        try:
            node = None
            for module in search_modules:
                try:
                    node = getattr(module, node_params["node_name"])
                    break
                except AttributeError:
                    continue
            if node is None:
                raise KeyError(f'Node function "{node_params["node_name"]}" '
                               f'not recognized')
            node(data_dict=data_dict, **node_params)
        except Exception:
            log.warning(
                f'Unhandled error during node {node_params["node_name"]}!')
            log.warning(traceback.format_exc())

    if save_figures and hlp_mod.get_param('figures', data_dict) is None:
        log.warning('save_figures is True but there are no figures to save.')
        save_figures = False
    if save_processed_data or save_figures:
        save_mod.Save(data_dict, save_processed_data=save_processed_data,
                      save_figures=save_figures, **params)