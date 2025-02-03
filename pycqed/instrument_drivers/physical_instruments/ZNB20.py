from qcodes.instrument_drivers.rohde_schwarz import ZNB


class ZNBChannel(ZNB.ZNBChannel):
    """
    Extends the qcodes ZNBChanel class by visa commands that were not
    implemented.
    """
    def __init__(
            self,
            parent: "ZNB",
            name: str,
            channel: int,
            **kwargs):
        """
        Calls parent init and adds more parameters
        """
        super().__init__(parent, name, channel, **kwargs)
        n = channel

        self.add_parameter(
            name="avg_mode",
            get_cmd=f"SENS{n}:AVERage:MODE?",
            set_cmd=self._set_avg_mode,
            val_mapping={
                "auto": "AUTO\n",
                "flatten": "FLAT\n",
                "reduce": "RED\n",
                "moving": "MOV\n",
            },
        )

    def _set_avg_mode(self, val):
        channel = self._instrument_channel
        self.write(f"SENSe{channel}:AVERage:MODE {val}")


class ZNB20(ZNB.ZNB):
    CHANNEL_CLASS = ZNBChannel


