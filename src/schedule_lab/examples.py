from __future__ import annotations

from .adapters import build_fjsp, build_fsp, build_hfsp, build_jsp
from .model import Problem


def example_problems() -> dict[str, Problem]:
    return {
        "jsp": build_jsp(
            [
                [("M1", 3), ("M2", 2), ("M3", 2)],
                [("M1", 2), ("M3", 1), ("M2", 4)],
                [("M2", 4), ("M3", 3)],
            ],
            problem_id="example-jsp",
        ),
        "fsp": build_fsp(
            [[3, 2, 4], [2, 4, 3], [4, 3, 2], [3, 5, 1]],
            problem_id="example-fsp",
        ),
        "fjsp": build_fjsp(
            [
                [[("M1", 3), ("M2", 4)], [("M2", 2), ("M3", 3)]],
                [[("M1", 2), ("M3", 3)], [("M2", 4), ("M3", 2)]],
                [[("M2", 2), ("M3", 4)], [("M1", 3), ("M3", 2)]],
            ],
            problem_id="example-fjsp",
        ),
        "hfsp": build_hfsp(
            [[3, 4, 2], [2, 5, 3], [4, 2, 4], [3, 3, 2]],
            [2, 2, 1],
            problem_id="example-hfsp",
        ),
    }
