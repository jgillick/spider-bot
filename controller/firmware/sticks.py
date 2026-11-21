"""Analog stick reading: oversampling, calibration, deadzone, and scaling to the HID range.

Each axis keeps three raw ADC readings: `lo` (fully one way), `centre` (at rest) and `hi`
(fully the other way). A reading is scaled against whichever half it falls in, so a stick
whose centre isn't mid-travel still reaches full scale both ways.
"""
import analogio

import config

ADC_MAX = 65535       # analogio values are 0..65535
REPORT_MAX = 32767    # HID axis range is -32767..32767


class Axis:
    def __init__(self, pin, invert):
        self._adc = analogio.AnalogIn(pin)
        self.invert = invert
        self.lo = 0
        self.centre = ADC_MAX // 2
        self.hi = ADC_MAX

    def read_raw(self):
        """Average of OVERSAMPLE ADC readings."""
        adc = self._adc
        total = 0
        for _ in range(config.OVERSAMPLE):
            total += adc.value
        return total // config.OVERSAMPLE

    def set_calibration(self, lo, centre, hi):
        self.lo, self.centre, self.hi = lo, centre, hi

    def widen_extents(self, raw):
        """Stretch lo/hi to include `raw`, so travel past the calibration still counts."""
        if raw < self.lo:
            self.lo = raw
        if raw > self.hi:
            self.hi = raw

    def to_report(self, raw):
        """Raw ADC reading -> -32767..32767, with deadzone and inversion applied."""
        self.widen_extents(raw)
        c = self.centre
        span = (self.hi - c) if raw >= c else (c - self.lo)
        v = (raw - c) / span if span > 0 else 0.0      # -1.0 .. 1.0

        dz = config.DEADZONE
        if abs(v) <= dz:
            return 0
        v = (v - dz if v > 0 else v + dz) / (1.0 - dz)  # rescale so the edge of the deadzone is 0
        v = max(-1.0, min(1.0, v))
        if self.invert:
            v = -v
        return int(v * REPORT_MAX)


class Sticks:
    """The four axes in HID report order: LX, LY, RX, RY."""

    def __init__(self):
        pins = (config.PIN_LX, config.PIN_LY, config.PIN_RX, config.PIN_RY)
        self.axes = [Axis(p, inv) for p, inv in zip(pins, config.INVERT)]
        self._values = [0, 0, 0, 0]   # reused every read; no allocation in the main loop

    def start(self, stored):
        """Set up for normal use. Call with the sticks at rest.

        Centres are re-measured now. Extents come from `stored` (a list of (lo, centre, hi)
        per axis) if there is a calibration, otherwise DEFAULT_HALF_RANGE either side.
        """
        centres = self.measure_centres()
        for axis, centre, cal in zip(self.axes, centres, stored or [None] * 4):
            if cal is None:
                half = config.DEFAULT_HALF_RANGE
                axis.set_calibration(max(0, centre - half), centre, min(ADC_MAX, centre + half))
                continue
            lo, stored_centre, hi = cal
            stored_half = min(stored_centre - lo, hi - stored_centre)
            if abs(centre - stored_centre) > config.CENTRE_SANITY * stored_half:
                centre = stored_centre   # a thumb is probably on the stick; trust the stored centre
            axis.set_calibration(lo, centre, hi)

    def measure_centres(self, samples=8):
        sums = [0, 0, 0, 0]
        for _ in range(samples):
            for i, axis in enumerate(self.axes):
                sums[i] += axis.read_raw()
        return [s // samples for s in sums]

    def begin_sweep(self):
        """Start a calibration: measure centres, collapse the extents onto them."""
        for axis, c in zip(self.axes, self.measure_centres()):
            axis.set_calibration(c, c, c)

    def sweep_step(self):
        for axis in self.axes:
            axis.widen_extents(axis.read_raw())

    def calibration(self):
        return [(a.lo, a.centre, a.hi) for a in self.axes]

    def read(self):
        """Current positions, -32767..32767 each. Returns the same list object every call."""
        values = self._values
        for i, axis in enumerate(self.axes):
            values[i] = axis.to_report(axis.read_raw())
        return values
