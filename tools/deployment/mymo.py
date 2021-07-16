import sys

from mo.mo.utils.versions_checker import check_python_version
if __name__ == "__main__":
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.mo.main import main
    from mo.mo.utils.cli_parser import get_all_cli_parser  # pylint: disable=no-name-in-module

    sys.exit(main(get_all_cli_parser(), None))