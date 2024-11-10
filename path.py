import pathlib

def relative_path(file_path: str, underscore_file) -> str:
    return (pathlib.Path(underscore_file).parent / file_path).as_posix()