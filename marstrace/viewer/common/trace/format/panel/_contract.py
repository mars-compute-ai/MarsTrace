import json


class PanelContract:
    """The values the trace panel and its renderer agree on: inks, sizes, levels and the token."""

    # scope inks, assigned by scope order and never cycled: a filter that hides a
    # scope must not repaint the survivors
    SCOPE_INKS = (
        (
            13,
            132,
            166,
        ),  # teal
        (
            192,
            74,
            44,
        ),  # brick
        (
            77,
            84,
            184,
        ),  # indigo
        (
            176,
            128,
            26,
        ),  # ochre
        (
            0,
            121,
            90,
        ),  # pine
        (
            192,
            90,
            155,
        ),  # plum
        (
            125,
            138,
            31,
        ),  # olive
    )
    NEUTRAL_INK = (
        107,
        114,
        120,
    )

    # identity lives in the hue, so every scope prints at the same density
    DENSITY = 0.82

    # the exposure stays greyscale; the measurement layer is the one place colour is allowed
    FILM = "#fdfdfd"
    GRID = "rgba(21,24,27,0.06)"
    TICK = "#6b7278"
    EDGE = "rgba(21,24,27,0.16)"
    RULE = "rgba(21,24,27,0.15)"
    RAIL = "rgba(21,24,27,0.13)"
    MARK = "#0f7d8c"
    MARK_FILL = "rgba(15,125,140,0.13)"

    AXIS_H = 26
    AXIS_W = 54
    PAD_TOP = 8
    MIN_ROW_PX = 2.2
    MIN_MARK_PX = 3
    MIN_STRIPE_PX = 2.5
    MIN_RULE_PX = 5
    MIN_RAIL_PX = 6
    DRAG_AXIS_PX = 5
    GRAB_PX = 8

    # how far one wheel unit zooms, as a factor of the window
    ZOOM_PER_WHEEL_UNIT = 0.0015

    # vertical granularities, finest first
    LANE_LEVELS = (
        "thread",
        "warp",
        "warpgroup",
        "block",
    )

    # the token a copied selection is, and the version this package writes
    SELECTION_TOKEN_PREFIX = "mtrace/1"
    SELECTION_TOKEN_KEYS = (
        "kernel",
        "t",
        "lane",
        "order",
        "scopes",
    )

    @classmethod
    def ink_css(cls, index: int) -> str:
        """The scope's ink at full strength, as its position in the run fixes it.

        Args:
            index: the scope's position in the run

        Returns:
            the ink as a CSS ``rgb()`` color; the neutral ink past the palette's end
        """
        ink = cls.SCOPE_INKS[index] if index < len(cls.SCOPE_INKS) else cls.NEUTRAL_INK
        return f"rgb({ink[0]},{ink[1]},{ink[2]})"

    @classmethod
    def to_json(cls) -> str:
        """The contract's values, as the renderer reads them.

        Returns:
            the contract as compact JSON
        """
        return json.dumps(
            obj={
                "SCOPE_INKS": [list(ink) for ink in cls.SCOPE_INKS],
                "NEUTRAL_INK": list(cls.NEUTRAL_INK),
                "DENSITY": cls.DENSITY,
                "FILM": cls.FILM,
                "GRID": cls.GRID,
                "TICK": cls.TICK,
                "EDGE": cls.EDGE,
                "RULE": cls.RULE,
                "RAIL": cls.RAIL,
                "MARK": cls.MARK,
                "MARK_FILL": cls.MARK_FILL,
                "AXIS_H": cls.AXIS_H,
                "AXIS_W": cls.AXIS_W,
                "PAD_TOP": cls.PAD_TOP,
                "MIN_ROW_PX": cls.MIN_ROW_PX,
                "MIN_MARK_PX": cls.MIN_MARK_PX,
                "MIN_STRIPE_PX": cls.MIN_STRIPE_PX,
                "MIN_RULE_PX": cls.MIN_RULE_PX,
                "MIN_RAIL_PX": cls.MIN_RAIL_PX,
                "DRAG_AXIS_PX": cls.DRAG_AXIS_PX,
                "GRAB_PX": cls.GRAB_PX,
                "ZOOM_PER_WHEEL_UNIT": cls.ZOOM_PER_WHEEL_UNIT,
                "LANE_LEVELS": list(cls.LANE_LEVELS),
                "TOKEN_PREFIX": cls.SELECTION_TOKEN_PREFIX,
            },
            separators=(
                ",",
                ":",
            ),
        )
