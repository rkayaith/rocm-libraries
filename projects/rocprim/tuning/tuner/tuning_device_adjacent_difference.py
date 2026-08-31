#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Optional, OrderedDict, Callable
import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range
"""
MAX_BLOCK_SIZE_X = 1024
MIN_BLOCK_SIZE_X = 64
BLOCK_SIZE_INC = 64

STARTING_IPT = [1, 2]
MIN_IPT = 4
MAX_IPT = 32
IPT_INC = 4


class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_adjacent_difference')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = list(range(MIN_BLOCK_SIZE_X, MAX_BLOCK_SIZE_X + 1, BLOCK_SIZE_INC))
        params['__ipt__'] = STARTING_IPT + list(range(MIN_IPT, MAX_IPT + 1, IPT_INC))
        return params

    def _get_restrictions(
        self, value_type: str, _: Optional[str] = None
    ) -> Callable[[dict], bool]:
        size = self.bytes_size // TYPE_CONFIGS[value_type].size
        element_size = TYPE_CONFIGS[value_type].size

        def validate(params):
            block_size = params['block_size_x']
            items_per_thread = params['__ipt__']

            # Total size constraint
            if block_size * items_per_thread > size:
                return False

            # Memory size constraint
            if block_size * items_per_thread * element_size > 65536:
                return False

            # Block size constraint
            if block_size > 1024:
                return False

            # Items per thread constraint - high items_per_threads don't perform well
            if items_per_thread >= block_size:
                return False

            # High items_per_threads on gfx1030 cause HSA_STATUS_ERROR_INVALID_ISA
            if params.get("arch_name") == "gfx1030" and items_per_thread > 28:
                return False

            return True

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""
        for val_type in COMMON_KEY_TYPES:
            self.tune_type(val_type)


if __name__ == "__main__":
    Tuner.cli()