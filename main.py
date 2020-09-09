from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
from numpy import ndarray
from PIL import Image


Channel = ndarray


@dataclass
class Pic:
    red: Channel
    blue: Channel
    green: Channel


@dataclass
class Offset:
    row: int
    col: int

    def __sub__(self, other: Offset) -> Offset:
        return Offset(self.row - other.row, self.col - other.col)


@dataclass
class Pixel:
    row: int
    col: int

    def __add__(self, other: Offset) -> Pixel:
        return Pixel(self.row + other.row, self.col + other.col)


def ssd(x: ndarray, y: ndarray) -> float:
    return np.sum((x - y) ** 2)


def initialize_pic(img: ndarray) -> Pic:
    excess = len(img) % 3
    img = img[:-excess]
    b, g, r = np.split(img, indices_or_sections=3)
    return Pic(red=r, blue=b, green=g)


def sub(img: ndarray, topleft: Pixel, rows: int, cols: int) -> ndarray:
    return img[topleft.row : topleft.row + rows, topleft.col : topleft.col + cols]


def calc_align_basic(
    reference_img: Channel, other: Channel, offset_window: int = 15
) -> Offset:
    """Give best offset for alignment of 2 channels."""
    scores = {}

    for r_off, c_off in product(range(-offset_window, offset_window + 1), repeat=2):

        shifted = np.roll(other, shift=(r_off, c_off), axis=(0, 1))

        scores[(r_off, c_off)] = ssd(reference_img, shifted)
    r_off, c_off = min(scores, key=lambda k: scores[k])
    return Offset(r_off, c_off)


def shift_2d_replace(data, dx, dy, constant=0):
    """Shifts the array in two dimensions while setting rolled values to
    constant.
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def merge_aligned(ref: Channel, other: Channel, offset: Offset):
    return ref + shift_2d_replace(
        other,
        dx=offset.col,
        dy=offset.col,
    )


if __name__ == "__main__":
    import os

    proj_root = Path("~/cs194-26/proj1").expanduser()
    os.chdir(proj_root)
    img_path = Path("data/tobolsk.jpg")

    img = np.array(Image.open(img_path))
    pic = initialize_pic(img)
    r_align = calc_align_basic(pic.blue, pic.red)
    g_align = calc_align_basic(pic.blue, pic.green)
    print(merge_aligned(pic.blue, pic.red, r_align))
    print(pic.red.shape)