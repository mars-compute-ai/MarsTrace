import re
from dataclasses import dataclass, field
from ._contract import PanelContract
from ._data import PanelLane


@dataclass
class SelectionToken:
    """A stretch of time over some rows, as the panel's right-click menu copied it."""

    kernel: str  # the kernel's tag, as `kernel_tag` writes it
    t0: int  # from the run's first interval start
    t1: int
    unit: str  # ns or cyc
    level: str  # the lane level the ids name
    list_id: list[int] = field(default_factory=list)
    order: str = "thread"
    list_scope: list[str] = field(default_factory=list)

    # the leading identifier of a kernel's name, which its tag keeps
    LEAD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    # the longest leading identifier a tag keeps
    LEAD_LENGTH_MAX = 24
    # the panel's digest: a 31-shift string hash over 32-bit signed arithmetic, of which the
    # tag keeps four hex digits
    DIGEST_SHIFT = 5
    DIGEST_MASK = 0xFFFFFFFF
    DIGEST_SIGNED_MAX = 0x7FFFFFFF
    TAG_MODULUS = 0x10000

    @staticmethod
    def kernel_tag(kernel: str) -> str:
        """A kernel's leading identifier and a 4-hex digest of its whole name.

        The digest is the panel's, computed over 32-bit signed arithmetic so the two
        sides agree on every name; a mangled name runs to hundreds of characters and
        no token should carry one.

        Args:
            kernel: the kernel's full (mangled) name

        Returns:
            the tag, as ``<identifier>#<4 hex digits>``
        """
        digest = 0
        for char in kernel:
            digest = (
                (digest << SelectionToken.DIGEST_SHIFT) - digest + ord(char)
            ) & SelectionToken.DIGEST_MASK
            if digest > SelectionToken.DIGEST_SIGNED_MAX:
                digest -= SelectionToken.DIGEST_MASK + 1
        match = SelectionToken.LEAD_PATTERN.search(kernel)
        lead = (match.group(0) if match else "kernel")[: SelectionToken.LEAD_LENGTH_MAX]
        return f"{lead}#{(digest & SelectionToken.DIGEST_MASK) % SelectionToken.TAG_MODULUS:04x}"

    @staticmethod
    def _ids_of(spec: str) -> list[int]:
        """The ids `0-3,12,20-23` names, in order."""
        list_id: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                id_low, id_high = part.split(
                    sep="-",
                    maxsplit=1,
                )
                list_id.extend(
                    range(
                        int(id_low),
                        int(id_high) + 1,
                    )
                )
            else:
                list_id.append(int(part))
        return list_id

    @classmethod
    def parse(cls, token: str) -> "SelectionToken":
        """One selection token as its parts.

        Args:
            token: the token as the panel's right-click menu copied it

        Returns:
            the token's parts

        Raises:
            ValueError: the token is malformed; the message names what is wrong with it
        """
        text = " ".join(token.strip().split())
        if not text.startswith(PanelContract.SELECTION_TOKEN_PREFIX + " "):
            raise ValueError(
                f"not a selection token: it must begin with {PanelContract.SELECTION_TOKEN_PREFIX!r}, "
                f"as the panel's right-click menu writes it"
            )
        map_field: dict[str, str] = {}
        for part in text.split(" ")[1:]:
            if "=" not in part:
                raise ValueError(f"token field {part!r} is not key=value")
            key, value = part.split(
                sep="=",
                maxsplit=1,
            )
            map_field[key] = value

        for key in (
            "kernel",
            "t",
            "lane",
        ):
            if key not in map_field:
                raise ValueError(f"token is missing {key}=")

        match = re.fullmatch(
            pattern=r"(-?\d+):(-?\d+)(ns|cyc)",
            string=map_field["t"],
        )
        if match is None:
            raise ValueError(f"token time range {map_field['t']!r} is not <t0>:<t1>ns")
        t0, t1, unit = int(match.group(1)), int(match.group(2)), match.group(3)

        level, _, ids = map_field["lane"].partition(":")
        if level not in PanelContract.LANE_LEVELS:
            raise ValueError(
                f"token lane level {level!r} is not one of {', '.join(PanelContract.LANE_LEVELS)}"
            )
        return cls(
            kernel=map_field["kernel"],
            t0=min(
                t0,
                t1,
            ),
            t1=max(
                t0,
                t1,
            ),
            unit=unit,
            level=level,
            list_id=cls._ids_of(ids),
            order=map_field.get(
                "order",   # key
                "thread",  # default
            ),
            list_scope=[
                s
                for s in map_field.get(
                    "scopes",  # key
                    "",        # default
                ).split(",")
                if s
            ],
        )

    def rows_of(
        self,
        list_tid: list[int],
        block_size: int,
        warp_size: int,
        warps_per_group: int,
    ) -> list[int]:
        """The thread ids of the rows a token names, out of the ones a run holds.

        A level's id is the launch's own -- the warp and the warpgroup count from the
        grid, not from inside a block -- which is what lets the panel's lane axis
        refine from block down to thread without a count going back up.

        Args:
            list_tid: the thread ids of the run's rows
            block_size: threads per block, for a block-level token
            warp_size: threads per warp, for a warp-level token
            warps_per_group: warps per warpgroup, for a warpgroup-level token

        Returns:
            the thread ids the token selects, in the run's order
        """
        set_id = set(self.list_id)
        list_row: list[int] = []
        for tid in list_tid:
            lane = PanelLane.of(
                tid=tid,
                block_size=block_size,
                warp_size=warp_size,
                warps_per_group=warps_per_group,
            )
            if lane.id_at(self.level) in set_id:
                list_row.append(tid)
        return list_row
