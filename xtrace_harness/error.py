class HarnessError(Exception):
    """A failure of the harness that the user can act on: a bad job directory, a refused request."""


class ManifestError(HarnessError):
    """The job manifest ``xtrace.toml`` is missing, malformed or names something absent."""


class PlatformError(HarnessError):
    """MarsCompute refused a request or answered with something the harness cannot read.

    Args:
        message: what failed
        status: the HTTP status, or None when no response arrived
        code: the platform's error code, or None when the response named none
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
