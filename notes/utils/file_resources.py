from importlib import resources
from pathlib import Path

from notes._exceptions import FileReadError


def get_resource_path(
        file_path: str,
        package: str,
        sub_module: str,
) -> Path:
    """
    Retrieves the path to a resource file within the package.
    This function attempts to first use `importlib.resources` for preferred
    access to resources within the package. It falls back to `pkg_resources`
    for compatibility with older environments.

    :param file_path:
    :param package:
    :param sub_module:
    :return:
    """
    resource_path = None
    try:
        resource_path = resources.files(package).joinpath(
            sub_module, file_path
        )

    except TypeError:
        try:
            # python < 3.9
            import pkg_resources

            resource_path = Path(
                pkg_resources.resource_filename(
                    package, f"{sub_module}/{file_path}"
                )
            ).resolve()

        except Exception as exc:
            raise FileReadError(
                "Error loading resource file using pkg_resources",
                str(resource_path),
            ) from exc

    if not resource_path.exists():
        raise FileReadError("Resource file not found", str(resource_path))

    return resource_path


if __name__ == '__main__':
    # 示例
    print(get_resource_path(file_path="config.toml", package="config", sub_module="tttt/settings"))
    print(get_resource_path(file_path="settings/config.toml", package="config", sub_module="tttt"))
    print(get_resource_path(file_path="config.toml", package="config", sub_module=""))
