import logging
import sys


class ClosingFileHandler(logging.FileHandler):
    """A handler to log to a file, closing the file after each written record.

    This allows windows users to read the log file content from outside
    the python kernel while the python kernel is still running.
    """

    def emit(self, record):
        """
        Emit a record (opening the stream if needed) and close the stream
        afterwards.
        """
        super().emit(record)
        self.close()


class LogNegatedFilter(logging.Filter):
    """Helper class to negate a logging filter.

    Used in function log_level.

    Attributes:
        negated: the constant value True to allow checking via getattr
            whether a filter is a negated filter.
    """
    negated = True

    def filter(self, record):
        return not super().filter(record)


def log_file_add(filename, module=None, level='DEBUG', **kw):
    """Covenience function for adding log file handlers

    Adds a new ClosingFileHandler to the logging system and configures it
    according to the provided args.

    Args:
        filename (str): the filename (including path) of the log file
        module (str or module or None): the module whose log messages should be
            stored in the log file. The default is None, which will store log
            messages from all modules. Note: the function assumes that the
            logger name is the same as the module name (common convention in
            pycqed).
        level (str): the minimum log level that a message must have to be
            stored in the log file (default: 'DEBUG'). Allowed are strings
            that are known levels of the logging package, typically
            'CRITICAL', 'ERROR', 'WARNING', 'INFO', and 'DEBUG'.
        **kw: keyword arguments for the ClosingFileHandler

    Examples:
        log_file_add(r'D:\test_all.log')
        log_file_add(r'D:\test_pycqed_info.log', 'pycqed', 'INFO')
        log_file_add(r'D:\test_pulsar.log', pulsar_mod)
    """
    if module is not None and not isinstance(module, str):
        module = module.__name__
    h = ClosingFileHandler(filename, **kw)
    h.setLevel(level)
    h.setFormatter(
        logging.Formatter(fmt='%(asctime)s ' + logging.BASIC_FORMAT))
    if module:
        h.addFilter(logging.Filter(module))
    logging.getLogger().addHandler(h)


def log_level(level=None, module=None):
    """Covenience function for setting/getting log levels of modules

    This function configures the log level for the screen output. Log levels of
    log files added with log_file_add are not affected.

    Args:
        level (str or None): a str to which the log level should be set or the
            default value None, in which case the current log level is shown.
            Allowed are strings that are known levels of the logging package,
            typically 'CRITICAL', 'ERROR', 'WARNING', 'INFO', and 'DEBUG'.
        module (str or module or None): the module whose log level should be
            configured. The default is None, which configures the log level of
            the root logger. Note: the function assumes that the logger name
            is the same as the module name (common convention in pycqed).

    Examples:
        log_level('INFO')  # set current root log handler level
        log_level('DEBUG', pulsar_mod)  # set special level for a module
        print(log_level())  # print current root log handler level
        print(log_level(module=pulsar_mod))  # print current level of a module
        log_level(0, pulsar_mod)  # remove special level for a module
        print(log_level(module=pulsar_mod))  # print current level of a module
    """
    root = logging.getLogger()
    if module is None:
        log = root
    else:
        log = logging.getLogger(
            module if isinstance(module, str) else module.__name__)
        root_handlers = [
            h for h in root.handlers
            if getattr(getattr(h, 'stream', None), 'name', None) == 'stderr']
    module_handlers = [
        h for h in log.handlers
        if getattr(getattr(h, 'stream', None), 'name', None) == 'stderr']
    if level is None:
        return logging.getLevelName(
            module_handlers[0].level if len(module_handlers)
            else logging.NOTSET)
    if module and level == logging.NOTSET and len(module_handlers):
        # delete the filter in the root handler and delete the module
        # handler(s)
        [root.handlers[0].removeFilter(f)
         for f in root.handlers[0].filters
         if f.name == log.name and getattr(f, 'negated', False)]
        [log.removeHandler(h) for h in module_handlers]
    elif len(module_handlers):
        # update the existing handler(s)
        [h.setLevel(level) for h in module_handlers]
    else:
        # add a module handler
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(fmt=logging.BASIC_FORMAT))
        h.setLevel(level)
        log.addHandler(h)
        if module:
            # add a filter in the root handler(s)
            [h.addFilter(LogNegatedFilter(log.name)) for h in root_handlers]

