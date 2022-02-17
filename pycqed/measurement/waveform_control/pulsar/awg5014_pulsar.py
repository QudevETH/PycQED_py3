class AWG5014Pulsar:
    """
    Defines the Tektronix AWG5014 specific functionality for the Pulsar class
    """
    _supportedAWGtypes = (Tektronix_AWG5014, VirtualAWG5014, )

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, AWG5014Pulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)

        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_enforce_single_element'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Group all the pulses on this AWG into "
                                     "a single element. Useful for making sure "
                                     "that the master AWG has only one waveform"
                                     " per segment.")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 4)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=4/(1.2e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           get_cmd=lambda: 256/(1.2e9)) # Can not be triggered
                                                        # faster than 210 ns.
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           get_cmd=lambda: 0)
        self.add_parameter('{}_precompile'.format(awg.name),
                           initial_value=False,
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('{}_delay'.format(awg.name), initial_value=0,
                           label='{} delay'.format(awg.name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in  time")
        self.add_parameter('{}_trigger_channels'.format(awg.name),
                           initial_value=[],
                           label='{} trigger channels'.format(awg.name),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name),
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(awg.name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

        group = []
        for ch_nr in range(4):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_analog_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            id = 'ch{}m1'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            id = 'ch{}m2'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
        # all channels are considered as a single group
        for name in group:
            self.channel_groups.update({name: group})

    def _awg5014_create_analog_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_offset_mode'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Enum('software', 'hardware'))
        offset_mode_func = self.parameters['{}_offset_mode'.format(name)]
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset',
                                                        offset_mode_func),
                           get_cmd=self._awg5014_getter(awg, id, 'offset',
                                                        offset_mode_func),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 2.25))
        self.add_parameter('{}_distortion'.format(name),
                            label='{} distortion mode'.format(name),
                            initial_value='off',
                            vals=vals.Enum('off', 'precalculate'),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                            label='{} distortion dictionary'.format(name),
                            vals=vals.Dict(),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Numbers(0., 1.), initial_value=0.5)
        self.add_parameter('{}_compensation_pulse_delay'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_gaussian_filter_sigma'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

    def _awg5014_create_marker_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'marker')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset'),
                           get_cmd=self._awg5014_getter(awg, id, 'offset'),
                           vals=vals.Numbers(-2.7, 2.7))
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(-5.4, 5.4))

    @staticmethod
    def _awg5014_setter(obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def s(val):
                    if offset_mode_func() == 'software':
                        obj.set('{}_offset'.format(id), val)
                    elif offset_mode_func() == 'hardware':
                        obj.set('{}_DC_out'.format(id), val)
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                        '{}'.format(offset_mode_func()))
            elif par == 'amp':
                def s(val):
                    obj.set('{}_amp'.format(id), 2*val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def s(val):
                    h = obj.get('{}_high'.format(id_raw))
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), val + h - l)
                    obj.set('{}_low'.format(id_raw), val)
            elif par == 'amp':
                def s(val):
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), l + val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _awg5014_getter(self, obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def g():
                    if offset_mode_func() == 'software':
                        return obj.get('{}_offset'.format(id))
                    elif offset_mode_func() == 'hardware':
                        return obj.get('{}_DC_out'.format(id))
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                         '{}'.format(offset_mode_func()))

            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['{}_amp'.format(id)] \
                                   .get_latest()/2
                    else:
                        return obj.get('{}_amp'.format(id))/2
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def g():
                    return obj.get('{}_low'.format(id_raw))
            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        h = obj.get('{}_high'.format(id_raw))
                        l = obj.get('{}_low'.format(id_raw))
                    else:
                        h = obj.parameters['{}_high'.format(id_raw)]\
                            .get_latest()
                        l = obj.parameters['{}_low'.format(id_raw)]\
                            .get_latest()
                    return h - l
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms,
                                        repeat_pattern, **kw)

        pars = {
            'ch{}_m{}_low'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_m{}_high'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_offset'.format(ch + 1) for ch in range(4)
        }
        old_vals = {}
        for par in pars:
            old_vals[par] = obj.get(par)

        packed_waveforms = {}
        wfname_l = []

        grp_has_waveforms = {f'ch{i+1}': False for i in range(4)}

        for element in awg_sequence:
            if awg_sequence[element] is None:
                continue
            metadata = awg_sequence[element].pop('metadata', {})
            if list(awg_sequence[element].keys()) != ['no_codeword']:
                raise NotImplementedError('AWG5014 sequencer does '
                                          'not support codewords!')
            chid_to_hash = awg_sequence[element]['no_codeword']

            if not any(chid_to_hash):
                continue  # no waveforms

            maxlen = max([len(waveforms[h]) for h in chid_to_hash.values()])
            maxlen = max(maxlen, 256)

            wfname_l.append([])
            for grp in [f'ch{i + 1}' for i in range(4)]:
                wave = (chid_to_hash.get(grp, None),
                        chid_to_hash.get(grp + 'm1', None),
                        chid_to_hash.get(grp + 'm2', None))
                grp_has_waveforms[grp] |= (wave != (None, None, None))
                wfname = self._hash_to_wavename((maxlen, wave))
                grp_wfs = [np.pad(waveforms.get(h, [0]),
                                  (0, maxlen - len(waveforms.get(h, [0]))),
                                  'constant', constant_values=0) for h in wave]
                packed_waveforms[wfname] = obj.pack_waveform(*grp_wfs)
                wfname_l[-1].append(wfname)
                if any([wf[0] != 0 for wf in grp_wfs]):
                    log.warning(f'Element {element} starts with non-zero '
                                f'entry on {obj.name}.')

        if not any(grp_has_waveforms.values()):
            for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
                obj.set('{}_state'.format(grp), grp_has_waveforms[grp])
            return None

        self.awgs_with_waveforms(obj.name)

        nrep_l = [1] * len(wfname_l)
        goto_l = [0] * len(wfname_l)
        goto_l[-1] = 1
        wait_l = [1] * len(wfname_l)
        logic_jump_l = [0] * len(wfname_l)

        filename = 'pycqed_pulsar.awg'

        awg_file = obj.generate_awg_file(packed_waveforms, np.array(wfname_l).transpose().copy(),
                                         nrep_l, wait_l, goto_l, logic_jump_l,
                                         self._awg5014_chan_cfg(obj.name))
        obj.send_awg_file(filename, awg_file)
        obj.load_awg_file(filename)

        for par in pars:
            obj.set(par, old_vals[par])

        time.sleep(.1)
        # Waits for AWG to be ready
        obj.is_awg_ready()

        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            obj.set('{}_state'.format(grp), 1*grp_has_waveforms[grp])

        hardware_offsets = 0
        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            cname = self._id_channel(grp, obj.name)
            offset_mode = self.get('{}_offset_mode'.format(cname))
            if offset_mode == 'hardware':
                hardware_offsets = 1
            obj.DC_output(hardware_offsets)

        return awg_file

    def _is_awg_running(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)

        return obj.get_state() != 'Idle'

    def _clock(self, obj, cid=None):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

    @staticmethod
    def _awg5014_group_ids(cid):
        """
        Returns all id-s corresponding to a single channel group.
        For example `Pulsar._awg5014_group_ids('ch2')` returns `['ch2',
        'ch2m1', 'ch2m2']`.

        Args:
            cid: An id of one of the AWG5014 channels.

        Returns: A list of id-s corresponding to the same group as `cid`.
        """
        return [cid[:3], cid[:3] + 'm1', cid[:3] + 'm2']

    def _awg5014_chan_cfg(self, awg):
        channel_cfg = {}
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            cid = self.get('{}_id'.format(channel))
            amp = self.get('{}_amp'.format(channel))
            off = self.get('{}_offset'.format(channel))
            if self.get('{}_type'.format(channel)) == 'analog':
                offset_mode = self.get('{}_offset_mode'.format(channel))
                channel_cfg['ANALOG_METHOD_' + cid[2]] = 1
                channel_cfg['ANALOG_AMPLITUDE_' + cid[2]] = amp * 2
                if offset_mode == 'software':
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = off
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = 0
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 0
                else:
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = 0
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = off
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 1
            else:
                channel_cfg['MARKER1_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER2_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER{}_LOW_{}'.format(cid[-1], cid[2])] = \
                    off
                channel_cfg['MARKER{}_HIGH_{}'.format(cid[-1], cid[2])] = \
                    off + amp
            channel_cfg['CHANNEL_STATE_' + cid[2]] = 0

        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            if self.get('{}_active'.format(awg)):
                cid = self.get('{}_id'.format(channel))
                channel_cfg['CHANNEL_STATE_' + cid[2]] = 1
        return channel_cfg

    def _get_segment_filter_userregs(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._get_segment_filter_userregs(obj)
        return []

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        if not isinstance(awg, AWG5014Pulsar._supportedAWGtypes):
            return super().sigout_on(ch, on)
        chid = self.get(ch + '_id')
        if f"{chid}_state" in awg.parameters:
            awg.set(f"{chid}_state", on)
        else:  # it is a marker channel
            # We cannot switch on the marker channel explicitly. It is on if
            # the corresponding analog channel is on. Raise a warning if
            # the state (of the analog channel) is different from the
            # requested state (of the marker channel).
            if bool(awg.get(f"{chid[:3]}_state")) != bool(on):
                log.warning(f'Pulsar: Cannot set the state of a marker '
                            f'channel. Call sigout_on for the corresponding '
                            f'analog channel {chid[:3]} instead.')
        return
