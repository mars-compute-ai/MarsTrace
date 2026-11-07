import argparse
import filecmp
import os
import shutil
import sys


class SchemaSync:
    """Vendors the Xtrace JSON Schemas that the platform's contracts export.

    Args:
        source_directory: ``packages/contracts/json-schema/xtrace`` of a MaaS checkout
        target_directory: the harness's ``xtrace_harness/schemas``
    """

    SOURCE_PATH = (
        "packages",
        "contracts",
        "json-schema",
        "xtrace",
    )
    TARGET_PATH = (
        "xtrace_harness",
        "schemas",
    )
    SUFFIX = ".schema.json"
    EXIT_STALE = 1

    def __init__(
        self,
        source_directory: str,
        target_directory: str,
    ) -> None:
        self._source_directory = source_directory
        self._target_directory = target_directory

    @staticmethod
    def list_schema(directory: str) -> list[str]:
        """The schema files of a directory.

        Args:
            directory: the directory

        Returns:
            their names, sorted; none when the directory is absent
        """
        if not os.path.isdir(directory):
            return []
        return sorted(name for name in os.listdir(directory) if name.endswith(SchemaSync.SUFFIX))

    def list_stale(self) -> list[str]:
        """The schema files whose vendored copy is missing, different or no longer exported.

        Returns:
            their names
        """
        list_source = SchemaSync.list_schema(self._source_directory)
        list_target = SchemaSync.list_schema(self._target_directory)
        list_stale = sorted(set(list_target) - set(list_source))
        for name in list_source:
            target = os.path.join(
                self._target_directory,
                name,
            )
            if (
                not os.path.isfile(target)
                or not filecmp.cmp(
                    os.path.join(
                        self._source_directory,
                        name,
                    ),
                    target,
                    shallow=False,
                )
            ):
                list_stale.append(name)
        return sorted(list_stale)

    def sync(self) -> list[str]:
        """Copy every exported schema and remove the vendored ones no longer exported.

        Returns:
            the names that changed
        """
        list_stale = self.list_stale()
        os.makedirs(
            name=self._target_directory,
            exist_ok=True,
        )
        set_source = set(SchemaSync.list_schema(self._source_directory))
        for name in list_stale:
            target = os.path.join(
                self._target_directory,
                name,
            )
            if name in set_source:
                shutil.copyfile(
                    src=os.path.join(
                        self._source_directory,
                        name,
                    ),
                    dst=target,
                )
            else:
                os.remove(target)
        return list_stale


def main() -> None:
    """Copy the schemas from a MaaS checkout, or with ``--check`` fail when the copy is stale."""
    parser = argparse.ArgumentParser(
        description="Vendor the Xtrace JSON Schemas of a MaaS checkout into xtrace_harness/schemas.",
    )
    parser.add_argument(
        "maas",
        help="the MaaS checkout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="change nothing; exit 1 when the vendored schemas differ from the checkout's",
    )
    args = parser.parse_args()
    source_directory = os.path.join(
        args.maas,
        *SchemaSync.SOURCE_PATH,
    )
    if not SchemaSync.list_schema(source_directory):
        print(
            f"no *{SchemaSync.SUFFIX} in {source_directory}; export them in the checkout first",
            file=sys.stderr,
        )
        sys.exit(SchemaSync.EXIT_STALE)
    schema_sync = SchemaSync(
        source_directory=source_directory,
        target_directory=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            *SchemaSync.TARGET_PATH,
        ),
    )
    if args.check:
        list_stale = schema_sync.list_stale()
        for name in list_stale:
            print(
                f"stale: {name}",
                file=sys.stderr,
            )
        sys.exit(SchemaSync.EXIT_STALE if list_stale else 0)
    for name in schema_sync.sync():
        print(f"synced {name}")


if __name__ == "__main__":
    main()
