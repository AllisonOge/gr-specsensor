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

class set_multiply_const_xx(gr.sync_block):
    """
    docstring for block set_multiply_const_xx
    """
    def __init__(self, constant: int, vlen: int):

        if vlen == 1:
            in_sig = [np.complex64,]
            out_sig = [np.complex64,]
        else:
            in_sig = [(np.complex64, vlen),]
            out_sig = [(np.complex64, vlen),]
        gr.sync_block.__init__(self,
            name="set_multiply_const_xx",
            in_sig=in_sig,
            out_sig=out_sig)
        
        self.constant = constant
        self.vlen = vlen

        # register message port
        self.message_port_register_in(pmt.intern("constant"))
        self.set_msg_handler(pmt.intern("constant"), self.handle_msg)

    def handle_msg(self, msg):
        if pmt.is_dict(msg) and pmt.dict_has_key(msg, pmt.intern("constant")):
            self.constant = pmt.to_python(msg)["constant"]

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        # <+signal processing here+>
        out[:] = in0 * self.constant
        return len(output_items[0])
