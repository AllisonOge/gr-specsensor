#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Allison Ogechukwu.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import queue
import sqlite3
from scipy import special
import numpy as np
from gnuradio import gr
import pmt
import datetime


class test_stats(gr.sync_block):
    """
    Compute the test stats of an energy detector and decide if
    signal is present or absent given sensitivity of threshold

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
                 signal_edges: tuple, save: bool, sqlite_path: str, table_name: str):

        gr.sync_block.__init__(self,
                               name="test_stats",
                               in_sig=[(np.float32, fft_len),
                                       (np.float32, fft_len)],
                               out_sig=None)

        self.signal_edges = signal_edges
        self.save = save
        self.fft_len = fft_len
        self.vlen = vlen
        self.sensitivity = sensitivity
        self.queue = queue.Queue(maxsize=vlen*2)
        self.acc_pointer = 0
        self.table_name = table_name
        self.sqlite_path = sqlite_path
        self.log = gr.logger("test_stats")
        self.first_run = True

        # register message output port
        self.message_port_name = "channel_state"
        self.message_port_register_out(pmt.intern(self.message_port_name))

    def connect_client(self):
        try:
            self.connect = sqlite3.connect(self.sqlite_path)
            self.cursor = self.connect.cursor()
        except:
            print("Failed to connect")

    def create_table(self):
        # create a table
        channels = ""
        for i in range(len(self.signal_edges)):
            channels += f"""chan_{i+1} integer not null,"""

        self.cursor.execute(f"""create table if not exists {self.table_name} (
            id integer primary key autoincrement,
            {channels}
            created_at datetime
         )""")

        self.connect.commit()

    def build_threshold(self, noise_est):
        """Computes the threshold of the energy detector using the formula

        threshold = noise_pow * (1 + np.sqrt(2/(self.vlen*self.fft_len))*qinv(self.sensitivity))

        This is the threshold formular for a welch periodogram-based energy detector.
        [1] Martínez, D.M. and Andrade, Á.G., 2013. Performance evaluation of Welch's 
            periodogram-based energy detection for spectrum sensing. IET Communications, 
            7(11), pp.1117-1125.

        Parameters
        ----------
            noise_est: (any) array of the noise power of the channels

        Returns
        -------
            threshold of type based on the input parameter
        """
        def qinv(prob_fa): return np.sqrt(2)*special.erfinv(1 - 2*prob_fa)
        threshold = noise_est * \
            (1 + np.sqrt(2/(self.vlen*self.fft_len))*qinv(self.sensitivity))
        return threshold

    def get_sensor_values(self):
        signal_acc = np.array([])
        noise_acc = np.array([])

        # accumulate the data
        while not self.queue.empty():
            signal_acc = np.append(signal_acc, self.queue.get())
            noise_acc = np.append(noise_acc, self.queue.get())

        # average the data
        signal_acc = np.mean(signal_acc.reshape(
            (-1, self.fft_len)), axis=0)
        noise_acc = np.mean(noise_acc.reshape(
            (-1, self.fft_len)), axis=0)

        # detect signal
        quotient = np.sum(self.signal_edges)
        test_stats = []
        noise = []
        for n, sig_edge in enumerate(self.signal_edges):
            start_idx = (sig_edge*self.fft_len//quotient)*n
            stop_idx = (sig_edge*self.fft_len//quotient)*(n+1)
            test_stats.append(np.sum([(self.fft_len/self.vlen) * p for p in
                                      signal_acc[start_idx:stop_idx]]))
            noise.append(
                np.sum([(self.fft_len/self.vlen) * p for p in noise_acc[start_idx:stop_idx]]))
        test_stats = np.array(test_stats)
        noise = np.array(noise)

        threshold = self.build_threshold(noise)
        decision = np.array(
            list(map(lambda d: 1 if d == True else 0, test_stats >= threshold)))

        timestamp = datetime.datetime.now()
        
        if self.save:

            if self.first_run:
                self.connect_client()
                self.create_table()
                self.first_run = False

            # save the decision
            channels = ""
            placeholders = ""
            for i in range(1, 1+len(self.signal_edges)):
                if i < len(self.signal_edges):
                    channels += f"chan_{i}, "
                    placeholders += "?, "
                else:
                    channels += f"chan_{i}"
                    placeholders += "?"

            try:
                query = f"""insert into {self.table_name} ({channels}, created_at) values ({placeholders}, ?)"""
                self.cursor.execute(query, [*list(map(int, decision)), timestamp])
                self.connect.commit()
            except Exception as err:
                pass

        # publish message
        message = {
            str(timestamp): list(map(int, decision))
        }
        PMT_msg = pmt.to_pmt(message)
        self.message_port_pub(pmt.intern(self.message_port_name), PMT_msg)


    def work(self, input_items, output_items):
        in0 = input_items[0]
        in1 = input_items[1]
        # <+signal processing here+>
        self.acc_pointer += 1
        try:
            self.queue.put(in0, block=False)
            self.queue.put(in1, block=False)
        except queue.Full:
            self.log.debug(
                "The accumulate queue is full, will try to requeue in the next work call")

        if self.acc_pointer == self.vlen:

            self.get_sensor_values()
            self.acc_pointer = 0

        return len(input_items[0]) + len(input_items[1])
