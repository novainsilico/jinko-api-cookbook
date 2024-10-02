#!/usr/bin/env python
import re
import sys
import json
import os
import argparse


def convert_notebook(file_path):
    """Converts a Jupyter notebook to a python script.

    Args:
        file_path (str): file path or "-"

    Returns:
      str[]: list of lines
    """
    output = os.popen("jupyter nbconvert --to script %s --stdout" % (file_path)).read()
    return output.splitlines()


def find_parameters(lines):
    """Update the variable declarations using the @param comments

    Args:
        lines (str[]): lines of a python file

    Returns:
      dict: array of parameters
    """
    param_pattern = re.compile(r"^[ \t]*#[ \t]*@param[ \t]+(\{.*\})")
    parameters = []
    for index, line in enumerate(lines):
        if param_pattern.match(line):
            json_content = param_pattern.match(line).group(1)
            param = json.loads(json_content)
            param["line"] = index
            parameters.append(param)
            continue
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
List parameters found in a Jupyter notebook as a JSON array
""".strip(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=str, help="notebook file path")
    args = parser.parse_args()
    lines = convert_notebook(args.input)
    parameters = find_parameters(lines)
    print(json.dumps(parameters, indent=2))
