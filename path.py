import pathlib

def relative_path(file_path: str) -> str:
    underscore_file = eval("__file__")
    return (pathlib.Path(underscore_file).parent / file_path).as_posix()