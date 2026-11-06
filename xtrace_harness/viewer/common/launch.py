class LaunchShape:
    """How a kernel launch is shaped: a grid of blocks, each a block of threads, along x, y and z."""

    @staticmethod
    def count_blocks(grid_dim: tuple[int, int, int]) -> int:
        """The blocks a launch's grid holds.

        Args:
            grid_dim: the launch's grid, in blocks along x, y and z

        Returns:
            the blocks across the whole grid
        """
        return grid_dim[0] * grid_dim[1] * grid_dim[2]

    @staticmethod
    def count_block_threads(block_dim: tuple[int, int, int]) -> int:
        """The threads one block of a launch holds.

        Args:
            block_dim: the launch's block, in threads along x, y and z

        Returns:
            the threads of one block
        """
        return block_dim[0] * block_dim[1] * block_dim[2]

    @staticmethod
    def count_threads(
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ) -> int:
        """The threads a launch runs.

        Args:
            grid_dim: the launch's grid, in blocks along x, y and z
            block_dim: the launch's block, in threads along x, y and z

        Returns:
            the threads across the whole grid
        """
        return LaunchShape.count_blocks(grid_dim) * LaunchShape.count_block_threads(block_dim)

    @staticmethod
    def count_warps(
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        warp_size: int,
    ) -> int:
        """The warps a launch holds, each block's last warp counted even when partly filled.

        Args:
            grid_dim: the launch's grid, in blocks along x, y and z
            block_dim: the launch's block, in threads along x, y and z
            warp_size: the threads of one warp

        Returns:
            the warps across the whole grid
        """
        warps_per_block = (LaunchShape.count_block_threads(block_dim) + warp_size - 1) // warp_size
        return LaunchShape.count_blocks(grid_dim) * warps_per_block


__all__ = ["LaunchShape"]
