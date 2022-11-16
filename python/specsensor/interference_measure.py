#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from .signal_detector import signal_detector
from .find_interference import find_interference

from gnuradio import gr


class interference_measure(gr.hier_block2):
    """
    docstring for block interference_measure
    """

    def __init__(self, fft_len: int, vlen: int, sensitivity: float, signal_edges: tuple):
        gr.hier_block2.__init__(self,
                                "interference_measure",
                                # Input signature
                                gr.io_signature(
                                    2, 2, gr.sizeof_gr_complex*1),
                                gr.io_signature(0, 0, 0))  # Output signature

        # define blocks
        self.specsensor_sd0 = signal_detector(
            fft_len, vlen, sensitivity, signal_edges, False)
        self.specsensor_sd1 = signal_detector(
            fft_len, vlen, sensitivity, signal_edges, False)
        self.specsensor_fi = find_interference()

        # Define blocks and connect them
        self.connect((self, 0), (self.specsensor_sd0, 0))
        self.msg_connect((self.specsensor_sd0, "channel_state"),
                         (self.specsensor_fi, "in0"))
        self.connect((self, 1), (self.specsensor_sd1, 0))
        self.msg_connect((self.specsensor_sd1, "channel_state"),
                         (self.specsensor_fi, "in1"))

    def get_interference(self):
        self.specsensor_fi.get_interference()
