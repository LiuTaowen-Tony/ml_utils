import pathlib

def relative_path(file_path: str, underscore_file: str) -> str:
    return (pathlib.Path(file_path).parent / underscore_file).as_posix()