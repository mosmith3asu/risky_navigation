import cProfile
import io
import json
import os
import pickle
import pstats
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy import nan



# I/O


def save_pickle(data, filename):
    with open(fix_filetype(filename, ".pickle"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(fix_filetype(filename, ".pickle"), "rb") as f:
        return pickle.load(f)



def save_dict_to_file(dic, filename):
    dic = dict(dic)
    with open(fix_filetype(filename, ".txt"), "w") as f:
        f.write(str(dic))


def load_dict_from_txt(filename):
    return load_dict_from_file(fix_filetype(filename, ".txt"))


def save_as_json(data, filename):
    with open(fix_filetype(filename, ".json"), "w") as outfile:
        json.dump(data, outfile)
    return filename


def load_from_json(filename):
    with open(fix_filetype(filename, ".json"), "r") as json_file:
        return json.load(json_file)


def iterate_over_json_files_in_dir(dir_path):
    pathlist = Path(dir_path).glob("*.json")
    return [str(path) for path in pathlist]


def fix_filetype(path, filetype):
    if path[-len(filetype) :] == filetype:
        return path
    else:
        return path + filetype


def generate_temporary_file_path(
    file_name=None, prefix="", suffix="", extension=""
):
    if file_name is None:
        file_name = str(uuid.uuid1())
    if extension and not extension.startswith("."):
        extension = "." + extension
    file_name = prefix + file_name + suffix + extension
    return os.path.join(tempfile.gettempdir(), file_name)


# Randomness
def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def is_iterable(obj):
    return isinstance(obj, Iterable)
