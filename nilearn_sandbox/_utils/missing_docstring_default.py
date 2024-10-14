"""Utility to find non-documented default value in docstrings.

The script support either a folder or file path as argument and write the results
 in the file `missing_docstring_default.md`.
"""

import ast
import re
import sys
from pathlib import Path

from numpydoc.docscrape import NumpyDocString

PUBLIC_ONLY = True
SKIP_TESTS = True

REPO_URL = "https://github.com/nilearn/nilearn/blob/main/"

# "masking.py#L25

def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def get_missing(docstring, default_args):
    """Return missing default values documentation.

    Returns
    -------
    missing: list[Tuple[str, str]]
        Parameters missing from the docstring. `(arg name, arg value)`.
    """
    doc = NumpyDocString(docstring)
    params = {param.name: param for param in doc["Parameters"]}

    missing = []
    for argname, argvalue in default_args.items():
        if f"%({argname})s" in params:
            # Skip the generation for templated arguments.
            continue
        elif argname not in params:
            missing.append((argname, argvalue))
        else:
            # Match any of the following patterns:
            # arg : type, default value
            # arg : type, default=value
            # arg : type, default: value
            # arg : type, Default value
            # arg : type, Default=value
            # arg : type, Default: value
            m = re.search(
                r"(default|Default)(\s|:\s|=)(\'|\")?"
                + re.escape(str(argvalue))
                + r"(\'|\")?",
                "".join(params[argname].type),
            )
            if not m:
                missing.append((argname, argvalue))
            
    return missing

def get_default_args(func):
    # args with default value
    default_args = {
        k.arg: ast.unparse(v)
        for k, v in zip(func.args.args[::-1], func.args.defaults[::-1])
    }
    # kwargs with default value
    default_args |= {
        k.arg: ast.unparse(v)
        for k, v in zip(
            func.args.kwonlyargs[::-1], func.args.kw_defaults[::-1]
        )
    }    
    return default_args

def construct_header(relative_filename):
    return f"## [{relative_filename}]({REPO_URL}{relative_filename})\n"

def construct_function_line(relative_filename, func):
    function_url = f"{REPO_URL}{relative_filename}#L{func.lineno}"
    return f"[{relative_filename}::{func.name} - **line: {func.lineno}**]({function_url})\n"

if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    
    if input_path.is_dir():
        filenames = input_path.rglob("*.py")
    elif input_path.is_file():
        filenames = [input_path]
    else:
        raise ValueError("Expected a directory or file as argument.")

    log_file = Path("missing_docstring_default.md")
    # log_file.touch()
    with log_file.open("w") as fout:
        for filename in filenames:

            if any(
                str(x) == "externals"  for x in filename.parents
            ):
                continue

            if SKIP_TESTS and (str(filename.stem).startswith("test") or str(
                filename.parent.stem
            ).startswith("tests")):
                continue

            if PUBLIC_ONLY and (str(filename.stem).startswith("_") or any(
                str(x).startswith("_") for x in filename.parents
            )):
                continue

            tree = parse_ast(filename)
            create_module_header = True
            if input_path.is_dir():
                relative_filename = filename.relative_to(input_path.parent)
            else:
                relative_filename = filename

            for func in top_level_functions(tree.body):
                if PUBLIC_ONLY and func.name.startswith("_"):
                    continue

                docstring = ast.get_docstring(func, clean=False)
                if not docstring:
                    if create_module_header:
                        create_module_header = False
                        fout.write(construct_header(relative_filename))
                    fout.write(construct_function_line(relative_filename, func))
                    fout.write("- [ ] No docstring detected.\n\n")
                    continue

                default_args = get_default_args(func)

                missing = get_missing(docstring, default_args)
                # Log arguments with missing default values in documentation.
                if missing:
                    if create_module_header:
                        create_module_header = False
                        fout.write(construct_header(relative_filename))
                    fout.write(construct_function_line(relative_filename, func))
                    fout.write("<br>Potential arguments to fix\n")
                    for k, v in missing:
                        fout.write(f"- [ ] `{k}` `Default={v}`\n")
                    fout.write("\n")

## TESTS

def function_without_missing(allow_empty=False):
    """foo.

    Parameters
    ----------
    allow_empty : :obj:`bool`, default=False
        Allow loading an empty mask (full of 0 values).
    """

def test_get_missing():
    tree = parse_ast(Path(__file__))
    for func in top_level_functions(tree.body):
        if func.name != "function_without_missing":
            continue
        docstring = ast.get_docstring(func, clean=False)
        default_args = get_default_args(func)
        missing = get_missing(docstring, default_args)
        assert len(missing) == 0

