#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import pmt
from gnuradio import gr
import datetime
import time
from .cs_methods import *

cs_methods = ["random", "next", "prev", "nextstate", "idletime"]

class cognitive_controller(gr.basic_block):
    """
    docstring for block cognitive_controller
    """
    def __init__(self, frequencies, cs_method="idletime", model_path=None, log_file=None):
        gr.basic_block.__init__(self,
            name="cognitive_controller",
            in_sig=None,
            out_sig=None)

        if cs_method not in cs_methods:
            raise ValueError(
                "Channel selection method should be any of the following:", cs_methods)

        if cs_method == "random":
            self.cs_method = RandomChannelSelection()

        if cs_method == "next" or cs_method == "prev":
            self.cs_method = NextOrPreviousChannelSelection(cs_method)

        if cs_method == "nextstate":
            if not model_path:
                raise ValueError("model path must be provided")
            self.cs_method = CS1(cs_method, nchannels=len(frequencies), model_path=model_path)

        if cs_method == "idletime":
            if not model_path:
                raise ValueError("model path must be provided")
            self.cs_method = CS2(cs_method, model_path=model_path)

        self.frequencies = frequencies
        self.log_file = log_file
        
        # register input message ports
        self.message_port_register_in(pmt.intern("channel_state"))
        self.set_msg_handler(pmt.intern("channel_state"), self.handle_msg)

        # register output message ports
        self.message_port_register_out(pmt.intern("trans_mode"))
        self.message_port_register_out(pmt.intern("freq"))

    def handle_msg(self, msg):
        # TODO: add checks
        message = pmt.to_python(msg)  
        channel_state = list(message.values())[0]
        # instantiate and run channel selection algorithm
        self.algorithm(channel_state)

    def algorithm(self, cs):
        # get the channel
        selected_channel = self.cs_method.select_channel(cs)
        # if not None
        if not selected_channel:
            return
        # instruct physical layer
        # publish messages to swtich frequency and
        # switch from sensor mode to transmission mode
        if self.log_file:
            # TODO: optimize logging
            with open(self.log_file, "w") as f:
                f.write(
                    f"Selected channel: {selected_channel} at {datetime.datetime.now()}")

        PMT_msg = pmt.from_bool(False)
        self.message_port_pub(pmt.intern("trans_mode"), PMT_msg)
        PMT_msg = pmt.to_pmt(dict({"freq": self.frequencies[selected_channel]}))
        self.message_port_pub(pmt.intern("freq"), PMT_msg)
        # transmit for a second
        time.sleep(1.0)
        PMT_msg = pmt.from_bool(True)
        self.message_port_pub(pmt.intern("trans_mode"), PMT_msg)


    def forecast(self, noutput_items, ninputs):
        # ninputs is the number of input connections
        # setup size of input_items[i] for work call
        # the required number of input items is returned
        #   in a list where each element represents the
        #   number of required items for each input
        ninput_items_required = [noutput_items] * ninputs
        return ninput_items_required

    def general_work(self, input_items, output_items):
        # For this sample code, the general block is made to behave like a sync block
        ninput_items = min([len(items) for items in input_items])
        noutput_items = min(len(output_items[0]), ninput_items)
        output_items[0][:noutput_items] = input_items[0][:noutput_items]
        self.consume_each(noutput_items)
        return noutput_items

