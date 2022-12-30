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
import sqlite3
from .cs_methods import *
from .evaluation_metrics import *

cs_methods = ["random", "next", "prev", "hoyhtya", "renewaltheory", "proposed"]


class cognitive_controller(gr.basic_block):
    """
    docstring for block cognitive_controller
    """

    def __init__(self, frequencies, db_path, cs_method="proposed", model_path=None, log_file=None):
        gr.basic_block.__init__(self,
                                name="cognitive_controller",
                                in_sig=None,
                                out_sig=None)
        
        cs_method = cs_method.lower()
        if cs_method not in cs_methods:
            raise ValueError(
                "Channel selection method should be any of the following:", cs_methods)

        if cs_method == "random":
            self.cs_method = RandomChannelSelection()

        if cs_method == "next" or cs_method == "prev":
            self.cs_method = NextOrPreviousChannelSelection(cs_method)

        if cs_method == "hoyhtya":
            self.cs_method = Hoyhtya(cs_method, db_path)
        
        if cs_method == "renewaltheory":
            self.cs_method = RenewalTheory(cs_method, db_path)

        if cs_method == "proposed":
            if not model_path:
                raise ValueError("model path must be provided")
            self.cs_method = CS(cs_method, db_path, model_path=model_path)

        self.frequencies = frequencies
        self.log_file = log_file

        # performance metrics
        self.measure_sr = MeasureSwitchRate(None)

        # register input message ports
        self.message_port_register_in(pmt.intern("channel_state"))
        self.set_msg_handler(pmt.intern("channel_state"), self.handle_msg)

        # register output message ports
        self.message_port_register_out(pmt.intern("trans_mode"))
        self.message_port_register_out(pmt.intern("command"))

    def handle_msg(self, msg):
        # TODO: add checks
        message = pmt.to_python(msg)
        channel_state = list(message.values())[0]
        # instantiate and run channel selection algorithm
        self.algorithm(channel_state)

    def algorithm(self, cs):
        # get the channel
        selected_channel = self.cs_method.select_channel(cs)
        self.measure_sr.count_switch_rate(selected_channel)
        # if None
        if selected_channel == None:
            return
        # instruct physical layer
        # publish messages to switch frequency and
        # switch from sensor mode to transmission mode
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(
                    f"Selected channel: {selected_channel} at {datetime.datetime.now()} \n")

        PMT_msg = pmt.from_bool(False)
        self.message_port_pub(pmt.intern("trans_mode"), PMT_msg)
        PMT_msg = pmt.to_pmt(
            dict({"freq": self.frequencies[selected_channel], "constant": 1}))
        self.message_port_pub(pmt.intern("command"), PMT_msg)
        # transmit for a 5 millisecond
        time.sleep(0.05)
        PMT_msg = pmt.from_bool(True)
        self.message_port_pub(pmt.intern("trans_mode"), PMT_msg)
        PMT_msg = pmt.to_pmt(
            dict({"constant": 0}))
        self.message_port_pub(pmt.intern("command"), PMT_msg)

        # flush message queue
        if not self.cs_method.name in ["random", "next"]:
            while not self.delete_head_nowait(pmt.intern("channel_state")) == None:
                pass

    def get_switch_rate(self):
        return self.measure_sr.get_switch_rate()

    def work(self, input_items, output_items):
        return len(input_items[0])
