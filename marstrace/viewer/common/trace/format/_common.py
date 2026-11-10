class TraceClock:
    """The clocks a trace reads its timestamps from, and the units a duration on each is shown in."""

    # the steps between the units a duration is scaled to
    THOUSAND = 1_000
    MILLION = 1_000_000
    BILLION = 1_000_000_000

    @staticmethod
    def format_ticks(
        ticks: int,
        clock_type: str = "gpu",
    ) -> str:
        """Human-readable duration in the unit its clock counts.

        The "gpu" and "anchor" clocks both carry nanoseconds -- the second is rebuilt
        onto the first's axis at decode -- while "sm" counts a core's own cycles, which
        no rate converts to time here and which are comparable only within one CTA.

        Args:
            ticks: the duration, in the clock's units
            clock_type: the clock the duration was read from: "gpu", "sm" or "anchor"

        Returns:
            the duration with a unit suffix scaled to its size
        """
        if clock_type == "sm":
            cycles = float(ticks)
            if cycles < TraceClock.THOUSAND:
                return f"{cycles:.0f} cyc"
            if cycles < TraceClock.MILLION:
                return f"{cycles / TraceClock.THOUSAND:.2f}k cyc"
            return f"{cycles / TraceClock.MILLION:.2f}M cyc"
        ns = float(ticks)
        if ns < TraceClock.THOUSAND:
            return f"{ns:.0f} ns"
        if ns < TraceClock.MILLION:
            return f"{ns / TraceClock.THOUSAND:.2f} µs"
        if ns < TraceClock.BILLION:
            return f"{ns / TraceClock.MILLION:.2f} ms"
        return f"{ns / TraceClock.BILLION:.2f} s"
