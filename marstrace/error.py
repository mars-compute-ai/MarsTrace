class MarsTraceError(Exception):
    """A failure the user can act on: a bad job directory, a refused request."""


class ManifestError(MarsTraceError):
    """The job manifest ``marstrace.toml`` is missing, malformed or names something absent."""


class ServiceError(MarsTraceError):
    """The MarsTrace service refused a request or answered with something the client cannot read.

    Args:
        message: what failed
        status: the HTTP status, or None when no response arrived
        code: the service's error code, or None when the response named none
        retryable: whether the service said the same request may succeed later
        request_id: the id the service logged the request under, or None
        add_credit_url: the console page where credit is added, on a refusal for credit
        retry_after_seconds: how long the service asked to wait before the same request, or
            None
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | None = None,
        retryable: bool = False,
        request_id: str | None = None,
        add_credit_url: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        self.add_credit_url = add_credit_url
        self.retry_after_seconds = retry_after_seconds

    def to_json(self) -> dict[str, str | int | bool | None]:
        """The error as a tool result carries it.

        Returns:
            the message, status, code, retryable flag, request id, credit page and wait
        """
        return {
            "message": str(self),
            "status": self.status,
            "code": self.code,
            "retryable": self.retryable,
            "request_id": self.request_id,
            "add_credit_url": self.add_credit_url,
            "retry_after_seconds": self.retry_after_seconds,
        }


class ToolArgumentError(MarsTraceError):
    """An MCP tool was called with arguments its input schema does not admit."""
