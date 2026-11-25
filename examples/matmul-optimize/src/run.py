import subprocess
import sys


class Benchmark:
    """Times the kernel at the sizes the job optimizes for, with the driver the build made."""

    DRIVER = "./build/matmul"
    LIST_SIZE = [
        "1024",
        "2048",
    ]

    @staticmethod
    def run() -> int:
        """Print each size's time in microseconds.

        Returns:
            the driver's exit status
        """
        return subprocess.run(
            [Benchmark.DRIVER, "time", *Benchmark.LIST_SIZE],
            check=False,
        ).returncode


if __name__ == "__main__":
    sys.exit(Benchmark.run())
