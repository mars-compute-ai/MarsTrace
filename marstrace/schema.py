import re
from typing import Any
from marstrace.error import ServiceError


class ResponseSchema:
    """The shapes of the MarsTrace service's answers, and the check every answer passes.

    A shape is a Python type (``str``, ``int``, ``bool``, ``dict``, ``list``), a compiled
    pattern a string must match, a set of allowed strings, ``None`` for JSON null, a tuple
    of alternatives, a one-item list of the shape of every item, a dict ``{str: shape}`` of
    a map's values, or a dict of required keys, where a key ending in ``?`` may be absent.
    Keys a shape does not name are accepted, so a service that adds fields stays readable.
    """

    OPTIONAL_SUFFIX = "?"
    # first characters of a departure that names a nested key or item
    SET_PATH_START = {
        ".",
        "[",
    }
    MONEY = re.compile(r"^(0|[1-9][0-9]*)$")
    SIGNED_MONEY = re.compile(r"^(0|-?[1-9][0-9]*)$")
    TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
    SHA256 = re.compile(r"^[0-9a-f]{64}$")
    # prefixed ids: the prefix, then 26 lowercase Crockford base32 characters
    ID_BODY = "[0-9abcdefghjkmnpqrstvwxyz]{26}"
    UPLOAD_ID = re.compile(rf"^upl_{ID_BODY}$")
    JOB_ID = re.compile(rf"^job_{ID_BODY}$")
    SET_JOB_STATUS = {
        "queued",
        "starting",
        "running",
        "finishing",
        "succeeded",
        "failed",
        "cancelled",
    }
    SET_ATTEMPT_STATUS = {
        "starting",
        "running",
        "finishing",
        "succeeded",
        "failed",
        "cancelled",
    }
    SET_FAILURE_CLASS = {
        "supplier",
        "platform",
        "workload",
        "cancel",
    }
    SET_UPLOAD_ROLE = {
        "cubin",
        "source_archive",
        "launch_manifest",
        "tensor",
        "file",
    }
    SET_QUALITY = {
        "usable",
        "partial",
        "unusable",
    }
    SET_CLOCK_TYPE = {
        "gpu",
        "sm",
        "anchor",
    }
    SET_GRANULARITY = {
        "thread",
        "warp",
    }
    ERROR = {
        "code": str,
        "message": str,
        "retryable": bool,
        "details?": {str: str},
    }
    CODE_MESSAGE = {
        "code": str,
        "message": str,
    }
    ENVELOPE = {
        "request_id": str,
        "data?": (dict, list, None),
        "error": (ERROR, None),
    }
    RATES = {
        "gpu_micro_usd_per_hour": MONEY,
        "job_fee_micro_usd?": MONEY,
        "agent_input_micro_usd_per_million": MONEY,
        "agent_cache_read_micro_usd_per_million": MONEY,
        "agent_cache_write_5m_micro_usd_per_million": MONEY,
        "agent_cache_write_1h_micro_usd_per_million": MONEY,
        "agent_output_micro_usd_per_million": MONEY,
    }
    AGENT_LIMITS = {
        "max_turns": int,
        "max_input_tokens": int,
        "max_output_tokens": int,
        "max_wall_seconds": int,
    }
    HARDWARE = {
        "hardware": [
            {
                "id": str,
                "gpu": str,
                "architecture": str,
                "available": bool,
                "service_model": str,
                "rates": (RATES, None),
            }
        ],
        "agent": {
            "model": str,
            "defaults": AGENT_LIMITS,
            "maximums": AGENT_LIMITS,
        },
    }
    CREDIT = {
        "billing_mode": str,
        "billing_status": str,
        "available_micro_usd": (SIGNED_MONEY, None),
        "held_micro_usd": MONEY,
        "minimum_micro_usd": MONEY,
        "admissible": bool,
        "add_credit_url": str,
    }
    UPLOAD_GRANT = {
        "upload_id": UPLOAD_ID,
        "expires_at": TIME,
        "upload_url?": str,
        "parts?": [
            {
                "part_number": int,
                "offset": int,
                "size_bytes": int,
                "upload_url": str,
            }
        ],
        "headers?": {str: str},
    }
    UPLOAD = {
        "upload_id": UPLOAD_ID,
        "role": SET_UPLOAD_ROLE,
        "name": str,
        "size_bytes": int,
        "sha256": SHA256,
        "content_type": str,
        "status": {"verified"},
        "expires_at": TIME,
    }
    QUOTE = {
        "service_model": str,
        "rate_card_version_id": str,
        "rates": RATES,
        "bounds_per_attempt": {
            "gpu_milliseconds": int,
            "agent_input_tokens": int,
            "agent_output_tokens": int,
        },
        "cost_at_bounds_micro_usd": {
            "job_fee?": MONEY,
            "gpu_per_attempt": MONEY,
            "agent_tokens_per_attempt": MONEY,
            "total": MONEY,
        },
        "minimum_micro_usd": MONEY,
        "available_micro_usd": (SIGNED_MONEY, None),
        "admissible": bool,
        "message?": str,
    }
    JOB = {
        "job_id": JOB_ID,
        "account_id": str,
        "status": SET_JOB_STATUS,
        "created_at": TIME,
        "started_at": (TIME, None),
        "ended_at": (TIME, None),
        "hardware": str,
        "agent": AGENT_LIMITS,
        "limits": {"max_attempts": int},
        "progress": (
            {
                "agent_turn": int,
                "gpu_seconds": int,
                "usage_micro_usd": MONEY,
            },
            None,
        ),
        "attempts": [
            {
                "number": int,
                "status": SET_ATTEMPT_STATUS,
                "failure": (SET_FAILURE_CLASS, None),
                "error": (CODE_MESSAGE, None),
                "started_at": TIME,
                "ended_at": (TIME, None),
            }
        ],
        "cost": {
            "held_micro_usd": (MONEY, None),
            "charged_micro_usd": (MONEY, None),
            "lines": [
                {
                    "dimension": str,
                    "quantity": int,
                    "amount_micro_usd": MONEY,
                }
            ],
        },
        "result": (
            {
                "name": str,
                "sha256": SHA256,
                "size_bytes": int,
                "download_url": str,
                "expires_at": TIME,
            },
            None,
        ),
        "error": (CODE_MESSAGE, None),
    }
    JOB_PAGE = {
        "items": [JOB],
        "next_cursor": (str, None),
    }
    RESULT = {
        "manifest": {
            "quality": SET_QUALITY,
            "reason_codes": [str],
            "file": {
                "name": str,
                "size_bytes": int,
                "sha256": SHA256,
            },
            "device": dict,
            "clock_type": SET_CLOCK_TYPE,
            "granularity": SET_GRANULARITY,
        },
        "download_url": str,
        "download_expires_at": TIME,
        "result_expires_at": TIME,
    }

    @staticmethod
    def matches(
        value: Any,
        shape: Any,
    ) -> str | None:
        """Where a value departs from a shape.

        Args:
            value: the decoded JSON value
            shape: the shape it should have

        Returns:
            a description of the first departure, or None when the value has the shape
        """
        if shape is None:
            return None if value is None else "is not null"
        if isinstance(
            shape,  # obj
            tuple,  # class_or_tuple
        ):
            list_departure = [
                ResponseSchema.matches(
                    value=value,
                    shape=alternative,
                )
                for alternative in shape
            ]
            if None in list_departure:
                return None
            return " or ".join(departure for departure in list_departure if departure is not None)
        if isinstance(
            shape,       # obj
            re.Pattern,  # class_or_tuple
        ):
            if (
                isinstance(
                    value,  # obj
                    str,    # class_or_tuple
                )
                and shape.match(value)
            ):
                return None
            return f"is not a string of the form {shape.pattern}"
        if isinstance(
            shape,  # obj
            set,    # class_or_tuple
        ):
            return None if value in shape else f"is not one of {', '.join(sorted(shape))}"
        if isinstance(
            shape,  # obj
            list,   # class_or_tuple
        ):
            if not isinstance(
                value,  # obj
                list,   # class_or_tuple
            ):
                return "is not an array"
            for item_index, item in enumerate(value):
                departure = ResponseSchema.matches(
                    value=item,
                    shape=shape[0],
                )
                if departure is not None:
                    return ResponseSchema._nested(
                        path=f"[{item_index}]",
                        departure=departure,
                    )
            return None
        if isinstance(
            shape,  # obj
            dict,   # class_or_tuple
        ):
            return ResponseSchema._matches_object(
                value=value,
                shape=shape,
            )
        if shape is int:
            is_integer = isinstance(
                value,  # obj
                int,    # class_or_tuple
            ) and not isinstance(
                value,  # obj
                bool,   # class_or_tuple
            )
            return None if is_integer else "is not an integer"
        if isinstance(
            value,  # obj
            shape,  # class_or_tuple
        ):
            return None
        return f"is not of type {shape.__name__}"

    @staticmethod
    def _nested(
        path: str,
        departure: str,
    ) -> str:
        """A departure of a nested value, prefixed with the path that reaches it."""
        separator = "" if departure[0] in ResponseSchema.SET_PATH_START else " "
        return f"{path}{separator}{departure}"

    @staticmethod
    def _matches_object(
        value: Any,
        shape: dict[Any, Any],
    ) -> str | None:
        """Where a value departs from an object shape or a map shape."""
        if not isinstance(
            value,  # obj
            dict,   # class_or_tuple
        ):
            return "is not an object"
        if str in shape:
            for key, item in value.items():
                departure = ResponseSchema.matches(
                    value=item,
                    shape=shape[str],
                )
                if departure is not None:
                    return ResponseSchema._nested(
                        path=f".{key}",
                        departure=departure,
                    )
            return None
        for shape_key, item_shape in shape.items():
            key = shape_key.removesuffix(ResponseSchema.OPTIONAL_SUFFIX)
            if key not in value:
                if shape_key.endswith(ResponseSchema.OPTIONAL_SUFFIX):
                    continue
                return f"has no '{key}'"
            departure = ResponseSchema.matches(
                value=value[key],
                shape=item_shape,
            )
            if departure is not None:
                return ResponseSchema._nested(
                    path=f".{key}",
                    departure=departure,
                )
        return None

    @staticmethod
    def check(
        value: Any,
        shape: Any,
        where: str,
    ) -> Any:
        """Check that an answer has its shape.

        Args:
            value: the decoded answer
            shape: the shape it should have
            where: the route that answered, for the message

        Returns:
            the value

        Raises:
            ServiceError: the answer departs from the shape
        """
        departure = ResponseSchema.matches(
            value=value,
            shape=shape,
        )
        if departure is not None:
            raise ServiceError(
                f"{where} answered with an unexpected body: "
                + ResponseSchema._nested(
                    path="the answer",
                    departure=departure,
                )
            )
        return value
