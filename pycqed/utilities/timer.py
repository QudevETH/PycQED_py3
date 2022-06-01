import numpy as np
import datetime as dt
import logging
import time
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory
from pycqed.measurement.hdf5_data import write_dict_to_hdf5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import functools


log = logging.getLogger(__name__)


class Timer(OrderedDict):
    HDF_GRP_NAME = "Timers"
    NAME_CKPT_START = "start"
    NAME_CKPT_END = "end"

    def __init__(self, name="timer", fmt="%Y-%m-%d %H:%M:%S.%f", name_separator=".",
                 verbose=False, auto_start=True, **kwargs):
        self.fmt = fmt
        self.name = name
        self.name_separator = name_separator
        self.verbose = verbose
        # timer should not start logging when initializing with previous values
        if len(kwargs):
            auto_start = False

        # initialize previous checkpoints
        for ckpt_name, values in kwargs.items():
            if isinstance(values, str):
                values = eval(values)
            try:
                self.checkpoint(ckpt_name, values=values, log_init=False)
            except Exception as e:
                log.warning(f'Could not initialize checkpoint {ckpt_name}. Skipping.')
        if auto_start:
            self.checkpoint(self.NAME_CKPT_START)

    @staticmethod
    def from_dict(timer_dict, **kwargs):
        kwargs.update(dict(auto_start=False))
        tm = Timer(**kwargs)
        for ckpt_name, values in timer_dict.items():
            tm.checkpoint(ckpt_name, values=values, log_init=False)
        return tm

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print(args)
            if hasattr(args[0], "timer"):
                args[0].timer.checkpoint(func.__qualname__ + self.name_separator + self.NAME_CKPT_START)
            else:
                log.warning(f'Using @Timer decorator on {args[0]} but {args[0]} has no .timer attribute.'
                            'Time will not be logged.')
            output = func(*args, **kwargs)
            if hasattr(args[0], "timer"):
                args[0].timer.checkpoint(func.__qualname__ + self.name_separator + self.NAME_CKPT_END)
            return output

        return wrapper

    def __enter__(self):
        if self.get(self.NAME_CKPT_START, None) is None:
            # overwrite auto_start because when used in "with" statement, start must happen at beginning
            self.checkpoint(self.NAME_CKPT_START)
        if self.verbose:
            lvl = log.level
            log.setLevel(logging.INFO)
            log.info(f'Start of {self.name}: {self[self.NAME_CKPT_START].get_start()}')
            log.setLevel(lvl)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.checkpoint(self.NAME_CKPT_END)

        if self.verbose:
            lvl = log.level
            log.setLevel(logging.INFO)
            log.info(f'End of {self.name}: {self[self.NAME_CKPT_END].get_end()}. Duration: {self.duration(return_type="str")}')
            log.setLevel(lvl)

    def checkpoint(self, name, values=(), log_every_x=1, log_init=True):
        if name not in self:
            self[name] = Checkpoint(name, values=values, fmt=self.fmt, log_every_x=log_every_x, log_init=log_init)
        elif len(values) != 0:
            self[name].extend(values)
        else:
            self[name].log_time()

    def duration(self, keys=None, return_type="seconds"):
        if keys is None:
            keys = (self.find_earliest()[0], self.find_latest()[0])
        try:
            duration = self[keys[1]].get("latest")[1] - self[keys[0]].get("earliest")[1]
            if return_type == "seconds":
                return duration.total_seconds()
            elif return_type == "time_delta":
                return duration
            elif return_type == "str":
                return str(duration)
            else:
                raise ValueError(f'return_type={return_type} not understood.')
        except KeyError as ke:
            log.error(f"Could not find key in timer: {ke}. Available keys: {self.keys()}")

    def find_keys(self, query="", mode='endswith'):
        """
        Finds keys of checkpoints based on query. Defaults to returning all keys.
        Args:
            query (str): string to search
            mode (str): 'exact', 'contains', 'startswith', 'endswith'.
                Decides which string matching function to use.

        Returns:
            matching_keys (list)
        """
        assert mode in ('exact', 'contains', 'startswith', 'endswith'), \
            f"Unknown mode: {mode}"
        matches = []
        for s in self:
            match_func = dict(exact=s.__eq__, contains=s.__contains__,
                              startswith=s.startswith, endswith=s.endswith)
            if match_func[mode](query):
                matches.append(s)
        return matches

    def find_earliest(self, after=None, checkpoints="all"):
        if checkpoints == "all":
            checkpoints = list(self)
        if after is None:
            after = dt.datetime(1900, 1, 1)
        earliest_val = ()
        for i, ckpt_name in enumerate(checkpoints):
            earliest_val_ckpt = self[ckpt_name].get("earliest")
            if len(earliest_val_ckpt):
                if after < earliest_val_ckpt[1] and len(earliest_val) == 0:
                    #initialize earliest value with encountered first value after "after"
                    earliest_val = (ckpt_name,) + earliest_val_ckpt
                if after < earliest_val_ckpt[1] < earliest_val[2]:
                    earliest_val = (ckpt_name,) + earliest_val_ckpt
        return earliest_val

    def find_latest(self, before=None):
        if before is None:
            before = dt.datetime(9999, 1, 1) # in case our code is still used in the year 9998.
        latest_val = ()
        for i, ckpt_name in enumerate(self):
            latest_val_ckpt = self[ckpt_name].get("latest") # (index in ckpt, value)
            if len(latest_val_ckpt):
                if before > latest_val_ckpt[1] and len(latest_val) == 0:
                    #initialize earliest value with encountered first value after "after"
                    latest_val = (ckpt_name,) + latest_val_ckpt
                if before > latest_val_ckpt[1] > latest_val[2]:
                    latest_val = (ckpt_name,) + latest_val_ckpt
        return latest_val

    def save(self, data_object, group_name=None):
        '''
        Saves metadata on the MC (such as timings)
        '''
        if group_name is None:
            group_name = self.name
        set_grp = data_object.create_group(group_name)
        d = {k: repr(v) for k, v in self.items()}
        write_dict_to_hdf5(d, entry_point=set_grp,
                               overwrite=False)

    def sort(self, sortby="earliest", reverse=False, checkpoints="all"):
        """
        Returns a sorted list of checkpoint names.
        Args:
            sortby (str): which element to retrieve from each checkpoint,
                passed to "which" in Checkpoint.get().
            reverse (bool): If False, sorts list in by increasing start time if True,
                sorts checkpoint list by decreasing start time end time.
        Returns:

        """
        times = {}
        if checkpoints == "all":
            checkpoints = list(self)
        else:
            for ckpt in checkpoints:
                assert ckpt in self, f"Checkpoint: {ckpt} not found in {self}"
        for ckpt in checkpoints:
            times[ckpt] = self[ckpt].get(sortby)[1]
        arg_sorted = sorted(range(len(list(times.values()))),
                            key=list(times.values()).__getitem__)
        return np.array(list(times))[arg_sorted[::-1] if reverse else arg_sorted]

    def find_start_end(self, ckpt_name, start_suffix=None,
                       end_suffix=None, assert_single_match=True):
        if start_suffix is None:
            start_suffix = self.name_separator + self.NAME_CKPT_START
        if end_suffix is None:
            end_suffix = self.name_separator + self.NAME_CKPT_END
        start = self.find_keys(ckpt_name + start_suffix, mode="exact")
        end = self.find_keys(ckpt_name + end_suffix, mode="exact")
        if assert_single_match:
            assert len(start) == 1, \
                f"Could not find unique start checkpoint name for " \
                f"{ckpt_name + self.name_separator + self.NAME_CKPT_START}: {start}"
            assert len(end) == 1, \
                f"Could not find unique end checkpoint name for " \
                f"{ckpt_name + self.name_separator + self.NAME_CKPT_END}: {end}"
            return (start[0], end[0])

        return (start, end)

    def get_ckpt_fragments(self, checkpoints="all"):
        """
        Combines checkpoints to extract fragments of timers and returns
        a dict of starting checkpoints and their duration. It tries to pair
        up checkpoints with same name but ending in ".start" and ".end".
        If it does find a pair, then the key in the return dict is the name
        of the start checkpoint without ".start", and the value is a list
        of tuples of the form (start_date0, duration0).
        If a checkpoint has an arbitrary name, then it is paired with itself
        and therefore the durations for each fragment will always be 0, but
        the starting time is still usefull to know when the checkpoint was
        triggered (possibly several times).
        Examples:
            >>> tm = Timer("mytimer", auto_start=False)
            >>> for _ in range(2):
            >>>     tm.checkpoint('mytimer.sleep.start')
            >>>     time.sleep(1)
            >>>     tm.checkpoint('mytimer.sleep.end')
            >>>     # unpaired checkpoint example:
            >>>     tm.checkpoint('mytimer.end_of_loop')
            >>> tm.get_ckpt_fragments()
            >>> {'mytimer.end_of_loop': [
            >>>     (dt.datetime(2021, 2, 2, 10, 1, 16, 178730), dt.timedelta(0)),
            >>>     (dt.datetime(2021, 2, 2, 10, 1, 17, 186281), dt.timedelta(0))],
            >>>  'mytimer.sleep': [
            >>>    (dt.datetime(2021, 2, 2, 10, 1, 15, 173375),
            >>>         dt.timedelta(0, 1, 5355)),
            >>>    (dt.datetime(2021, 2, 2, 10, 1, 16, 178730),
            >>>         dt.timedelta(0, 1, 7551))]}

        Args:
            checkpoints:

        Returns:

        """
        if checkpoints == "all":
            end_ckpts = self.find_keys(self.name_separator +
                                       self.NAME_CKPT_END,
                                         mode="endswith")
            ckpts = [ckpt for ckpt in self if ckpt not in end_ckpts]
            ckpts = self.sort(checkpoints=ckpts)
        elif np.ndim(checkpoints):
            ckpts = []
            for ckpt in checkpoints:
                ckpt_start = self.find_keys(ckpt + self.name_separator +
                                            self.NAME_CKPT_START,
                               mode="contains")
                ckpts.extend(ckpt_start)
        else:
            raise NotImplementedError(f"checkpoint mode : {checkpoints} not "
                                 f"implemented")
        ckpt_pairs = []
        for ckpt in ckpts:
            if ckpt.endswith(self.name_separator + self.NAME_CKPT_START):
                ckpt = ckpt[:-len(
                self.name_separator + self.NAME_CKPT_START)]
                ckpt_pairs.append(self.find_start_end(ckpt))
            elif ckpt.endswith(self.name_separator + self.NAME_CKPT_END):
                ckpt = ckpt[:-len(
                    self.name_separator + self.NAME_CKPT_END)]
                ckpt_pairs.append(self.find_start_end(ckpt))
            else:
                # checkpoint not part of "start" or "end" binome,
                # will return twice same checkpoint in pair
                ckpt_pairs.append(self.find_start_end(ckpt,
                                                      start_suffix="",
                                                      end_suffix=""))

        all_start_and_durations = {}
        for i, (s, e) in enumerate(ckpt_pairs):
            start_and_durations = []
            for s_value, e_value in zip(self[s], self[e]):
                if e_value < s_value:
                    log.warning(
                        f'Checkpoint {s}: End time: {e_value} occurs before start'
                        f' time: {s_value}. Skipping.')
                    continue
                start_and_durations.append((s_value, e_value - s_value))
            if s.endswith(self.name_separator + self.NAME_CKPT_START):
                s = s[:-len(self.name_separator + self.NAME_CKPT_START)]
            all_start_and_durations[s] = start_and_durations

        return all_start_and_durations

    def plot(self, checkpoints="all", type="bar", fig=None, ax=None, bar_width=0.45,
             xunit='min', xlim=None, date_format=None, annotate=True, title=None,
             time_axis=False, alpha=None, show_sum="absolute", ax_kwargs=None,
             tight_layout=True, milliseconds=False):
        """
        Plots a timer as a timeline or broken horizontal bar chart.
        Args:
            checkpoints:
            type (str): "bar" or "timeline". "bar" produces a broken horizontal
                bar plot where each checkpoint is on a separate row. "timeline"
                produces a single line timeline where checkpoints are of
                different colors.
            fig (Figure):
            ax (Axis):
            bar_width (float):
            xunit (string): unit for x axis; ['us', 'ms', 's', 'min', 'h',
                'day', 'week']
            xlim (tuple): convenience argument for the x axis, in unit of xunit
            date_format:
            annotate (bool):
            title (str):
            time_axis (bool): whether or not to include an additional x axis to
                display the dates.
            alpha:
            show_sum (str): "absolute", "relative" or None. If "absolute",
                shows the cumulative time for each checkpoint in absolute time,
                "relative" shows it in relative time of the first and last
                checkpoint in the timer.
            ax_kwargs (dict): additional kwargs for the axes properties (
                overwrite labels, scales, etc.).
            tight_layout (bool):
            milliseconds (bool): whether or not to include milliseconds in
                displayed total durations.

        Returns:

        """
        from pycqed.analysis_v3 import plotting as plot_mod
        unit_to_t_factor = dict(us=1e-6, ms=1e-3, s=1, min=60, h=3600,
                                day=3600 * 24, week=3600 * 24 * 7)
        all_start_and_durations = self.get_ckpt_fragments(checkpoints)
        total_durations = {ckpt_name: np.sum([t[1] for t in times])
                           for ckpt_name, times in all_start_and_durations.items()}

        total_durations_rel = {n: v.total_seconds() / self.duration() if
                               self.duration() != 0 else 1 for n, v in
                               total_durations.items() }

        # plotting
        if ax_kwargs is None:
            ax_kwargs = dict()
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(plot_mod.FIGURE_WIDTH_2COL, 2.104))
        if fig is None:
            fig = ax.get_figure()
        y_ticklabels = []
        ref_time = self.find_earliest()[-1]
        t_factor = unit_to_t_factor.get(xunit, xunit)
        for i, (label, values) in enumerate(all_start_and_durations.items()):
            i = -i  # such that the labels appear in the order specified by
                    # all_start_and_duration from top to bottom, which is the most i
                    # intuitive ordering if a specific order is provided.
            values = [((v[0] - ref_time).total_seconds() / t_factor,
                       v[1].total_seconds() / t_factor) for v in values]

            if type == "bar":
                ax.broken_barh(values, ((i - bar_width), bar_width * 2), color="C2",
                               label=label, alpha=alpha, edgecolor="C2",
                               linewidth=0.1)
                tform = blended_transform_factory(ax.transAxes, ax.transData)
                if show_sum == "relative":
                    ax.annotate(f"{total_durations_rel[label] * 100:05.2f} %",
                                (1.01, i), xycoords=tform)
                if show_sum == "absolute":
                    ax.annotate(
                        self._human_delta(total_durations[label],
                                          milliseconds=milliseconds) + " ",
                        (1.01, i), xycoords=tform)
                y_ticklabels.append(label)

            elif type == "timeline":
                if alpha is None:
                    alpha = 0.1
                l = " " + label.split(self.name_separator)[-1]
                [ax.plot([v[0], v[0]], [0, 1], label=l if v == values[0] else None,
                         color=f"C{np.abs(i)}") for v in values]

                [ax.fill_betweenx([0,1], v[0], v[0] + v[1], alpha=0.1,
                                  edgecolor=None,
                                  color=f"C{np.abs(i)}") for v in values]
                [ax.plot([v[0] + v[1]]*2, [0, 1], color=f"C{np.abs(i)}") for v in values]

        if time_axis:
            xmin, xmax = ax.get_xlim() if xlim is None else xlim
            ax_time = ax.twiny()
            ax_time.set_xlim(ref_time + dt.timedelta(seconds=xmin * t_factor),
                             ref_time + dt.timedelta(seconds=xmax * t_factor), )
            ax_time.set_xlabel("Time, $t$")
            if date_format is not None:
                ax_time.set_major_formatter(mdates.DateFormatter(date_format))
            fig.autofmt_xdate(ha="left")

        if annotate:
            if type == "bar":
                ax.set_yticks(-np.arange(len(all_start_and_durations)))
                ax.set_yticklabels(y_ticklabels)
            else:
                ax.legend(frameon=False, fontsize="x-small")
        if title is None:
            title = self.name
        ax.set_title(title)
        ax.set_xlabel(f"Duration, $d$ ({xunit})")

        if xlim is not None:
            ax_kwargs.update(dict(xlim=xlim))
        ax.set(**ax_kwargs)

        if tight_layout:
            fig.tight_layout()
        return fig

    def table(self, checkpoints="all"):
        """
        Table representation of the duration stored in a timer.
        Args:
            checkpoints:

        Returns:

        """
        import pandas as pd
        all_start_and_durations = self.get_ckpt_fragments(checkpoints)

        total_durations = {ckpt_name: np.sum([t[1] for t in times]) for ckpt_name, times
                           in
                           all_start_and_durations.items()}
        total_durations_rel = {n: v.total_seconds() / self.duration() for n, v in
                               total_durations.items()}
        df = pd.DataFrame([total_durations, total_durations_rel])
        df = df.T
        df.columns = ['Absolute cumulated time', "Relative time"]
        return df

    @staticmethod
    def _human_delta(tdelta, milliseconds=False):
        """
        Takes a timedelta object and formats it for humans.
        Usage:
            # 149 day(s) 8 hr(s) 36 min 19 sec
            print human_delta(datetime(2014, 3, 30) - datetime.now())
        Example Results:
            23 sec
            12 min 45 sec
            1 hr(s) 11 min 2 sec
            3 day(s) 13 hr(s) 56 min 34 sec
        :param tdelta: The timedelta object.
        :return: The human formatted timedelta
        """
        d = dict(days=tdelta.days)
        d['hrs'], rem = divmod(tdelta.seconds, 3600)
        d['min'], d['sec'] = divmod(rem, 60)
        if milliseconds:
            d['msecs'] = int(np.round(tdelta.microseconds * 1e-3))
        else:
            d['sec'] += int(np.round(tdelta.microseconds * 1e-6))

        if d['days'] != 0:
            fmt = '{days} day(s) {hrs:02}:{min:02}:{sec:02}'
        else:
            fmt = '{hrs:02}:{min:02}:{sec:02}'
        if milliseconds:
            fmt += '.{msecs:03}'
        return fmt.format(**d)


def multi_plot(timers, **plot_kwargs):
    """
    Plots several timers in a single plot. Combines the checkpoints of different
    timers into a single timer
    Args:
        timers (list):
        **plot_kwargs:

    Returns:

    """
    # create dummy timer that contains checkpoints of other timers
    # note: this won't work if several timers have the same checkpoint names
    tm = Timer(auto_start=False)
    [tm.update(t) for t in timers]
    return tm.plot(**plot_kwargs)


class Checkpoint(list):
    def __init__(self, name, values=(), log_every_x=1, fmt="%Y-%m-%d %H:%M:%S.%f",
                 min_timedelta=0, verbose=False, log_init=True):
        super().__init__()
        self.name = name
        self.fmt = fmt
        self.log_every_x = log_every_x
        self.counter = 0
        self.min_timedelta = min_timedelta
        self.verbose = verbose

        for v in values:
            if isinstance(v, str):
                v = dt.datetime.strptime(v, self.fmt)
            self.append(v)
        if log_init:
            self.log_time()

    def get_start(self):
        return self[0]

    def get_end(self):
        return self[-1]

    def get(self, which="latest"):
        """
        Convenience function to get lastest, earliest of nth checkpoint value (and index).
        Returns empty tuple if self is empty.
        Args:
            which (str or int): "earliest" returns value with earliest datetime,
                "latest" with the latest datetime, and integer n returns the nth datetime value
                 (starting from earliest)

        Returns:
            (index, datetime_value)

        """
        if which == "latest":
            which_ind = -1
        elif which == "earliest":
            which_ind = 0
        elif isinstance(which, int):
            which_ind = which
        else:
            raise ValueError(f'which not in ("latest", "earliest", integer): {which}')
        argsorted = np.argsort(self)[::-1] # from recent to latest
        if len(argsorted):
            return argsorted[which_ind], self[which_ind]
        else:
            return ()

    def active(self):
        if len(self) > 0 and \
                (dt.datetime.now() - self[-1]).total_seconds() < self.min_timedelta:
            #             (dt.datetime.now() - dt.datetime.strptime(self[-1], self.fmt)).total_seconds() < self.min_timedelta:

            return False
        else:
            return True

    def log_time(self, value=None):
        if self.active():
            if self.counter % self.log_every_x == 0:
                if value is None:
                    value = dt.datetime.now()  # .strftime(self.fmt)
                self.counter += 1
                self.append(value)

    def duration(self, ref=None, return_type="seconds"):
        if ref is None:
            ref = self[0]
        duration = self[-1] - ref
        if return_type == "seconds":
            return duration.total_seconds()
        elif return_type == "time_delta":
            return duration
        elif return_type == "str":
            return str(duration)
        else:
            raise ValueError(f'return_type={return_type} not understood.')

    #     def __enter__(self):
    #         if self.verbose:
    #             lvl = log.level
    #             log.setLevel(logging.INFO)
    #             log.info(f'Start of checkpoint {self.name}: {self[0]}.')
    #             log.setLevel(lvl)
    # #         self.log_time()
    #         return self

    #     def __exit__(self, exc_type, exc_val, exc_tb):
    #         self.log_time()

    #         if self.verbose:
    #             lvl = log.level
    #             log.setLevel(logging.INFO)
    #             log.info(f'End of checkpoint {self.name}: {self[-1]}. Duration: {self.duration(return_type="str")}')
    #             log.setLevel(lvl)

    def __str__(self):
        return "['" + "', '".join(dt.datetime.strftime(pt, self.fmt) for pt in self) + "']"

    def __repr__(self):
        return self.__str__()


class WatchdogException(Exception):
    """Exception raised by :class:`WatchdogTimer` when its timer expires."""
    pass


class WatchdogTimer:
    """Watchdog timer that can be used to make sure operation are not too long.

    Attributes:
        timeout: Timeout in seconds before the watchdog timer expires.
        mode: Watchdog mode: ``"raise"`` or ``"check"``. In raise mode, calling
            :meth:`check` will raise a :class:`WatchdogException`, while in
            check mode the function merely returns a boolean.
        error_msg: Optional error message of the raised exception.

    Example:

        import time
        from pycqed.utilities.timer import WatchdogTimer, WatchdogException

        try:
            with WatchdogTimer(2) as timer:
                for i in range(10):
                    print(i)
                    if i <= 2:
                        timer.reset()
                    time.sleep(0.5)
                    timer.check()
        except WatchdogException:
            print("WatchdogException was raised.")
    """

    ALLOWED_MODES = ["raise", "check"]
    DEFAULT_ERROR = "Watchdog timer has expired."

    def __init__(self, timeout:float, mode:str="raise", error_msg:str=None):

        if error_msg is None:
            error_msg = self.DEFAULT_ERROR
        self.timeout = timeout
        self.mode = mode
        self.start = 0
        self.error_msg = error_msg

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset(self):
        self.start = time.time()

    def check(self) -> bool:
        """Check if the timer has expired."""
        expired = time.time() - self.start > self.timeout
        if expired:
            if self.mode == "raise":
                raise WatchdogException(self.error_msg)
            else:
                return True
        else:
            return False