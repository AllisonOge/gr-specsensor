#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
import pmt

class multiplexer(gr.sync_block):
    """
    docstring for block multiplexer
    """
    def __init__(self, nselectors=2):
        gr.sync_block.__init__(self,
            name="multiplexer",
            in_sig=[np.complex64, ] * nselectors,
            out_sig=[np.complex64, ])
        
        self.nselectors = nselectors
        self.selector = 0

        # register a message input port
        self.message_port_name = "selector"
        self.message_port_register_in(pmt.intern(self.message_port_name))
        self.set_msg_handler(pmt.intern(self.message_port_name), self.handle_msg)

    def handle_msg(self, msg):
        self.selector = pmt.to_long(msg)


    def work(self, input_items, output_items):
        in_ = input_items[0: self.nselectors]
        out = output_items[0]
        # <+signal processing here+>
        out[:] = in_[self.selector]
        return len(output_items[0])
