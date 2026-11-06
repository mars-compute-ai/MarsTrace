class TraceSchema:
    """What every trace layout shares: the launch it covers and the buffer it fills.

    A subclass fixes the record layout, then sets ``stride_bytes``, ``total_threads`` and,
    for a warp-level run, ``warp_mode``, ``num_warps`` and ``warp_slab_bytes``.
    """

    # the binary units a size is printed in, below the largest
    LIST_BYTE_UNIT = (
        "B",
        "KiB",
        "MiB",
        "GiB",
    )
    BYTE_UNIT_LARGEST = "TiB"
    BYTE_UNIT_STEP = 1024

    BITS_PER_BYTE = 8
    # the widths a recorded value or the base it is read from takes
    VALUE_WIDTH_NARROW = 32
    VALUE_WIDTH_WIDE = 64
    # the records one thread (or warp) may write per site when the caller names no budget
    PER_THREAD_RECORDS_DEFAULT = 32

    # the threads of a warp, the shift that turns a thread index into its warp's, and the
    # warps of a warpgroup
    WARP_SIZE = 32
    WARP_SHIFT = WARP_SIZE.bit_length() - 1
    WARPS_PER_GROUP = 4
    # what the backend calls the group of threads that issue together
    WARP_LABEL = "warp"

    # the boundary kinds a scope record carries, by their reserved ids: stable across runs,
    # so an instrumented kernel encodes them as compile-time constants
    MAP_TYPE_ID_RESERVED = {
        "SCOPE_START": 1,
        "SCOPE_END": 2,
        "POINT": 3,
    }

    stride_bytes: int
    total_threads: int
    warp_mode: bool
    num_warps: int
    warp_slab_bytes: int

    @classmethod
    def format_bytes(cls, size: float) -> str:
        """A size in the largest binary unit that keeps it below one step.

        Args:
            size: the size, in bytes

        Returns:
            the size with its unit: whole bytes, else one decimal place
        """
        for unit in cls.LIST_BYTE_UNIT:
            if size < cls.BYTE_UNIT_STEP:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
            size /= cls.BYTE_UNIT_STEP
        return f"{size:.1f}{cls.BYTE_UNIT_LARGEST}"

    @property
    def total_bytes(self) -> int:
        """Total bytes the trace buffer must hold.

        Per-warp blocks in warp mode, per-thread slabs otherwise.
        """
        if self.warp_mode:
            return self.num_warps * self.warp_slab_bytes
        return self.total_threads * self.stride_bytes
