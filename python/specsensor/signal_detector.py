#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#



from gnuradio.fft import window
from gnuradio import gr, blocks, analog, fft
from .test_stats import test_stats

class signal_detector(gr.hier_block2):
    """
    A signal detector that determines the state of the channel
    within the sensed band given the signal edges

    Parameters
    ----------
        fft_len: (int) length of FFT
        vlen: (int) length of vector stream
        sensitivity: (float) a value between 0 and 1 to determine the sensitivity
        of the signal detector to the noise floor. Higher values mean lower threshold
        sqlite_path: (string) path to the sqlite database
        table_name: (str) name of the table to save the data

    Attributes
    ----------
    """
    def __init__(self, fft_len: int, vlen: int, sensitivity: float, 
                    signal_edges: tuple, sqlite_path: str, table_name: str="Sensor"):
        gr.hier_block2.__init__(self,
            "signal_detector",
            gr.io_signature(1, 1, gr.sizeof_gr_complex*1),  # Input signature
            gr.io_signature(0, 0, 0)) # Output signature

        self.fft_len = fft_len
        self.vlen = vlen

        self.message_port_name = "channel_state"
        self.message_port_register_hier_out(self.message_port_name)
        
        self.fft_fft_0 = fft.fft_vcc(fft_len, True, window.blackmanharris(fft_len), True, 1)
        self.fft_fft_1 = fft.fft_vcc(fft_len, True, window.blackmanharris(fft_len), True, 1)
        self.blocks_c2m_squared_0 = blocks.complex_to_mag_squared(fft_len)
        self.blocks_c2m_squared_1 = blocks.complex_to_mag_squared(fft_len)
        self.blocks_s2v_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_len)
        self.blocks_s2v_1 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_len)
        self.analog_noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, 1, 0)
        self.test_stats = test_stats(fft_len, vlen, sensitivity, 
                                            signal_edges, sqlite_path, table_name)

        # Define blocks and connect them
        self.connect(self, (self.blocks_s2v_0, 0))
        self.connect((self.blocks_s2v_0, 0), (self.fft_fft_0, 0))
        self.connect((self.fft_fft_0, 0), (self.blocks_c2m_squared_0, 0))
        self.connect((self.blocks_c2m_squared_0, 0), (self.test_stats, 0))
        self.connect((self.analog_noise_source, 0), (self.blocks_s2v_1, 0))
        self.connect((self.blocks_s2v_1, 0), (self.fft_fft_1, 0))
        self.connect((self.fft_fft_1, 0), (self.blocks_c2m_squared_1, 0))
        self.connect((self.blocks_c2m_squared_1, 0), (self.test_stats, 1))
        self.msg_connect(self.test_stats, "channel_state", self, "channel_state")
