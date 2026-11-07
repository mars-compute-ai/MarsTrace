class HarnessError(Exception):
    """A failure of the harness that the user can act on: a bad job directory, a refused request."""


class ManifestError(HarnessError):
    """The job manifest ``xtrace.toml`` is missing, malformed or names something absent."""


class PlatformError(HarnessError):
    """The Control Plane refused a request or answered with something the harness cannot read.

    Args:
        message: what failed
        status: the HTTP status, or None when no response arrived
        code: the platform's error code, or None when the response named none
        correlation_id: the id the platform logged the request under, or None
        top_up_url: the console page where credit is added, on ``insufficient_credit``
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | None = None,
        correlation_id: str | None = None,
        top_up_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.correlation_id = correlation_id
        self.top_up_url = top_up_url

    def to_json(self) -> dict[str, str | int | None]:
        """The error as a tool result carries it.

        Returns:
            the message, status, code, correlation id and top-up page
        """
        return {
            "message": str(self),
            "status": self.status,
            "code": self.code,
            "correlation_id": self.correlation_id,
            "top_up_url": self.top_up_url,
        }


class ToolArgumentError(HarnessError):
    """An MCP tool was called with arguments its input schema does not admit."""
