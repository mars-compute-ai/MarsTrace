import subprocess
import sys


class Verification:
    """Checks the kernel against a CPU reference, with the driver the build made."""

    DRIVER = "./build/matmul"
    # 1000 is not a multiple of the block, so the edge threads are checked too
    LIST_SIZE = [
        "256",
        "1000",
    ]

    @staticmethod
    def run() -> int:
        """Compare every element of the kernel's product with the CPU's.

        Returns:
            0 when every size matches, 1 otherwise
        """
        return subprocess.run(
            [Verification.DRIVER, "check", *Verification.LIST_SIZE],
            check=False,
        ).returncode


if __name__ == "__main__":
    sys.exit(Verification.run())
