from qcodes.instrument.base import Instrument as QcodesInstrument


class Instrument(QcodesInstrument):
    """
    Class for all QCodes instruments.
    """

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {'driver': str(self.__class__), 'name': self.name}
    