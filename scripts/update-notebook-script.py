#!/usr/bin/env python
import re
import sys
import json
import os
import argparse


def read_lines_from_file(file_path):
    """Read a file and return its content as a list of lines.

    Args:
        file_path (str): file path or "-"

    Returns:
      str[]: list of lines
    """
    if file_path == "-":
        return sys.stdin.readlines()
    with open(file_path, "r") as file:
        return file.readlines()


def write_lines_to_file(file_path, lines):
    """Write a list of lines to a file.

    Args:
        file_path (str): file path or "-"
        lines (str[]): list of lines
    """
    if file_path == "-":
        sys.stdout.writelines(lines)
        return
    with open(file_path, "w") as file:
        file.writelines(lines)


def insert_imports(lines):
    """Insert import statement after the shebang or before the first import statement

    Args:
        lines (str[]): lines of a python file

    Returns:
      str[]: lines with import statement inserted
    """
    imports = (
        """
from IPython.display import display
""".strip()
        + "\n"
    )
    result = []
    inserted = False
    for _, line in enumerate(lines):
        if inserted:
            result.append(line)
            continue
        if line.startswith("#!"):
            result.append(line)
            result.append(imports)
            inserted = True
            continue
        if line.startswith("import ") or line.startswith("from "):
            result.append(imports)
            result.append(line)
            inserted = True
            continue
        result.append(line)
    return result


def update_variable_declarations(lines, debug=False):
    """Update the variable declarations using the @param comments

    Args:
        lines (str[]): lines of a python file
        debug (bool): if True output extra messages to stderr. Defaults to False

    Returns:
      str[]: lines with variable declatations updated
    """
    param_pattern = re.compile(r"^[ \t]*#[ \t]*@param[ \t]+(\{.*\})")
    variable_declaration_pattern = re.compile(r"^([ \t]*\w+[ \t]*=[ \t]*)(.*)")
    current_param = None
    current_param_index = None
    result = []
    for index, line in enumerate(lines):
        if param_pattern.match(line):
            json_content = param_pattern.match(line).group(1)
            if current_param:
                sys.stderr.write(
                    "Parameter %s declared at line #%d was not used\n"
                    % (json.dumps(current_param), current_param_index)
                )
                continue
            current_param = json.loads(json_content)
            current_param_index = index
            if debug:
                sys.stderr.write(
                    "New parameter %s found at line #%d\n"
                    % (json.dumps(current_param), index)
                )
            if not ("type" in current_param):
                current_param["type"] = "string"
            result.append(line)
            continue
        elif current_param is not None:
            if variable_declaration_pattern.match(line):
                left = variable_declaration_pattern.match(line).group(1)
                right = os.getenv(current_param["name"])
                if right is not None:
                    if current_param["type"] == "string":
                        right = '"%s"' % (right)
                    updated_line = left + right
                    if debug:
                        sys.stderr.write(
                            "Variable declaration replaced at line #%d\n" % (index)
                        )
                        sys.stderr.write("  - from: %s\n" % (line.strip()))
                        sys.stderr.write("  - to: %s\n" % (updated_line.strip()))
                    line = updated_line
                else:
                    sys.stderr.write(
                        "No environment variable found for parameter %s used at line #%d\n"
                        % (json.dumps(current_param), index)
                    )
                result.append(line)
                current_param = None
                continue
        result.append(line)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
List parameters found in a Jupyter notebook as a JSON array
""".strip(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", default="-", type=str, help="input file path (use - for stdin)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        type=str,
        help="output file path (use - for stdout)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="if set, output extra information to stderr",
    )
    args = parser.parse_args()
    lines = read_lines_from_file(args.input)
    lines = insert_imports(lines)
    lines = update_variable_declarations(lines, debug=args.debug)
    write_lines_to_file(args.output, lines)
