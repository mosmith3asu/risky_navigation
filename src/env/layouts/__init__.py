import os


LAYOUTS_DIR = os.path.join(os.path.dirname(__file__))


def read_layout_dict(layout_name):
    return load_dict_from_file(
        os.path.join(LAYOUTS_DIR, layout_name + ".layout")
    )


def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())
