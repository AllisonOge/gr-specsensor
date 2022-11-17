#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import pmt
from gnuradio import gr

class find_interference(gr.basic_block):
    """
    Computes the interference as the number of interference / total number of calls
    """

    def __init__(self):
        gr.basic_block.__init__(self,
                                name="cognitive_controller",
                                in_sig=None,
                                out_sig=None)

        self.channel_state0 = []
        self.channel_state1 = []
        self.interference = 0
        self.count_calls = 0

        # register input message ports
        self.message_port_register_in(pmt.intern("in0"))
        self.set_msg_handler(pmt.intern("in0"), self.handle_msg0)

        self.message_port_register_in(pmt.intern("in1"))
        self.set_msg_handler(pmt.intern("in1"), self.handle_msg1)


    def handle_msg0(self, msg):
        # TODO: add checks
        message = pmt.to_python(msg)
        self.channel_state0 = list(message.values())[0]
        

    def handle_msg1(self, msg):
        # TODO: add checks
        message = pmt.to_python(msg)
        self.channel_state1 = list(message.values())[0]
        self.find_interference()
        

    def find_interference(self):
        # [0, 0, 1, 1] and [1, 0, 0, 0] == no interference
        # [1, 0, 1, 1] and [1, 0, 0, 0] == interference
        self.count_calls += 1
        for i, val in enumerate(self.channel_state0):
            if val == self.channel_state1[i]:
                self.interference += 1
        

    def get_interference(self):
        if self.count_calls > 0:
            return self.interference / self.count_calls
        else:
            return 0

    def work(self, input_items, output_items):
        return len(input_items[0])
