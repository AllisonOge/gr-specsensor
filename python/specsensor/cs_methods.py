import random
import tensorflow.lite as tflite
import os
import numpy as np
import sqlite3
from .utils import start_and_idle_time, on_and_off_time


class CSMethods:
    def __init__(self, name):
        self.name = name
        self.selected_channel = None

    def select_channel(self, channel_state):
        pass


class ML(CSMethods):
    """ML Parent Class to prepare the data and load model
       Attributes
       ----------
            dataset: (Array-like) saves the latest channel state within window size
            name: (string) name of the model
            window_size:(int) length of channel state to store in memory (default=10)
            model_interpreter: tflite interpreter to load model

        Methods
        -------
            update_dataset(channel_state)

    """

    def __init__(self, name: str, db_path: str, window_size: int = 10, model_path: str = os.path.abspath("./models/model.tflite")):
        self.db_path = db_path
        self.dataset: list = []
        self.window_size: int = window_size
        CSMethods.__init__(self, name)
        self.model_interpreter = tflite.Interpreter(model_path=model_path)
        # allocate memory
        self.model_interpreter.allocate_tensors()

    def update_dataset(self) -> None or True:
        """Read database for updated data

           Parameters
           ----------
                None

            Return 
            ------
                None or True
        """
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        try:
            dataset = cursor.execute(
                f"""select * from (select * from {self.name} order by id desc limit 10) order by id asc""").fetchall()
        except sqlite3.OperationalError:
            return
        if not len(dataset) > 0:
            return
        self.dataset = [[bit for i, bit in enumerate(row) if type(
            bit) is not str and i != 0] for row in dataset]
        con.close()
        return True

    def prepare_dataset(self) -> None:
        """Prepare data"""
        pass

    def get_prediction(self, channel_state) -> None or dict:
        """Prepare data and invoke model inference

           Parameters
           ----------
                channel_state: (1-D Array-like) state of the channel

            Return
            ------
                None or predictions
        """
        # update dataset and reshape to window size
        if self.update_dataset() is None:
            return
        if not len(self.dataset) >= self.window_size:
            return

        self.prepare_dataset()
        self.dataset = self.dataset[len(self.dataset)-self.window_size:]

        # set input tensor for the model
        X = np.array([r for r in self.dataset], dtype=np.float32).reshape(
            (-1, self.window_size, len(channel_state)))

        idx = self.model_interpreter.get_input_details()[0]["index"]
        # set input tensor
        self.model_interpreter.set_tensor(idx, X)
        # invoke predictions
        self.model_interpreter.invoke()

        idx = self.model_interpreter.get_output_details()[0]["index"]
        return self.model_interpreter.get_tensor(idx)


class RandomChannelSelection(CSMethods):
    """Random Channel Selection Method
       A channel selection method that chooses a channel from the 
       list of free channels randomly

       Attributes
       ----------
            selected_chan: (int) channel selected by the algorithm
            name: (string) name of the algorithm

       Methods
       -------
            select_channel(channel_state: int) get the selected channel by the algorithm

    """

    def __init__(self):
        CSMethods.__init__(self, "random")

    def select_channel(self, channel_state: list):
        """Randomly select a channel from the list of free channels
           Given the state of the channel generate the free channels
           and select a free channel randomly

           Parameters
           ----------
                channel_state: (Array-like) Array of ones and zeros 
                to show the present state of channel

            Return
            ------
                selected channel
        """
        if all(channel_state):
            return None

        free_channels = [i for i, state in enumerate(
            channel_state) if state == 0]
        # stay on selected channel if free
        if self.selected_channel in free_channels:
            return self.selected_channel
        else:
            self.selected_channel = random.choice(free_channels)
            return self.selected_channel

    def get_selected_channel(self):
        return self.selected_channel

    def set_selected_channel(self, selected_channel):
        self.selected_channel = selected_channel


class NextOrPreviousChannelSelection(CSMethods):
    """Next or Previous Channel Selection
       A channel selection algorithm that selects the next/previous free channel
       from the list of free channels

       Attributes
       ----------
            selected_chan: (int) channel selected by the algorithm
            name: (string) name of the algorithm

       Methods
       -------
            select_channel(channel_state: int) get the selected channel by the algorithm
    """

    def __init__(self, state="next"):
        if state not in ["next", "prev"]:
            raise Exception("State can only be next or previous")
        CSMethods.__init__(self, state)
        self.state = state

    def select_channel(self, channel_state):
        """Select the next or previous free channel from the list of 
           free channels

           Parameters
           ----------
                channel_state: (Array-like) Array of ones and zeros 
                to show the present state of channel

            Return
            ------
                selected channel
        """
        if all(channel_state):
            return None

        free_channels = []
        for i, state in enumerate(channel_state):
            if state == 0:
                free_channels.append(i)
        # print("Free channels:", free_channels)
        if not self.selected_channel:
            # if no selected channel return first channel
            return free_channels[0]
        # stay on selected channel if free
        if self.selected_channel in free_channels:
            return self.selected_channel
        else:
            if self.state == "next":
                if (self.selected_channel + 1) > len(channel_state) - 1:
                    # if next frequency is beyond length of free channels return the first
                    return free_channels[0]
                for chan in free_channels:
                    if chan >= self.selected_channel + 1:
                        return chan
                return free_channels[0]

            elif self.state == "prev":
                if (self.selected_channel - 1) < 0:
                    # if next frequency is beyond length of free channels return the first
                    return free_channels[len(free_channels)-1]
                for chan in free_channels[::-1]:
                    if chan <= self.selected_channel + 1:
                        return chan
                    return free_channels[len(free_channels)-1]

    def get_selected_channel(self):
        return self.selected_channel

    def set_selected_channel(self, selected_channel):
        self.selected_channel = selected_channel


class CS(ML):
    """Channel Selection Algorithm

       Make inference on the idle time of the channel 
       and select channel with highest idle time

       Attributes
       ----------
            dataset: (Array-like) saves the latest channel state within window size
            window_size:(int) length of channel state to store in memory (default=10)
            model_interpreter: tflite interpreter to load model
            selected_channel: (int) channel selected by algorithm

        Methods
        -------
            update_dataset()

    """

    def prepare_dataset(self):
        """ Compute the idletimes of the channel states sequence
            stored in dataset and updates it

            This method transforms the bits of ones and zeros to
            the idle time at each time slot e.g., a sequence of 
            [1, 0, 0, 0, 1, 0, 1] has the idle time representation
            of [0, 3, 2, 1, 0, 1, 0]

            Given the channel states tensor it transforms it to the
            corresponding idle times

            Parameters
            ----------
              None

            Returns
            -------
              None
        """
        idle_times = []
        for i in range(len(self.dataset)):
            idle_times.append([j[0][1] if len(j) > 0 and i+j[0][0] <= i else 0
                               for j in list(map(start_and_idle_time, np.transpose(self.dataset[i:])))])
        self.dataset = [r for r in idle_times]

    def select_channel(self, channel_state) -> None or int:
        """Select a channel with highest idle time prediction

           Parameters
           ----------
                channel_state: (1-D array-like) state of the channel

            Returns
            -------
                None or selected_channel

        """
        channel_state = list(channel_state)
        if all(channel_state):
            return
        # get predictions
        preds = self.get_prediction(channel_state)
        if preds is None:
            return

        preds = np.array(preds).flatten()

        # find free channels
        free_channels = [i for i, s in enumerate(channel_state) if s == 0]
        # maintain the selected channel if free
        if self.selected_channel in free_channels:
            return self.selected_channel

        # find channel with highest idle time
        max_preds = max([val for i, val in enumerate(
            preds) if channel_state[i] == 0])

        self.selected_channel = [
            i for i, val in enumerate(preds) if val == max_preds][0]
        return self.selected_channel


class Hoyhtya(CSMethods):
    """Implement Hoyhtya channel selection algorithm"""

    def __init__(self, name, db_path) -> None:
        CSMethods.__init__(self, name)
        self.idle_times = []
        self.db_path = db_path

    def update_idletimes(self):
        """Update the median idletimes of the channels"""
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        try:
            dataset = cursor.execute(
                f"""select * from {self.name}""").fetchall()
        except sqlite3.OperationalError:
            return
        if not len(dataset) > 0:
            return
        dataset = [[bit for i, bit in enumerate(row) if type(
            bit) is not str and i != 0] for row in dataset]
        self.idle_times = [np.median([idletime for s, idletime in idletimes]) if len(
            idletimes) > 0 else 0 for idletimes in list(map(start_and_idle_time, np.array(dataset).transpose()))]
        return True

    def select_channel(self, channel_state):
        if all(channel_state):
            return
        if not self.update_idletimes():
            return
        max_idletime = np.max(
            [idletime for i, idletime in self.idle_times if channel_state[i] == 0])
        if max_idletime == 0:
            return
        return [i for i, idletime in enumerate(self.idle_times) if idletime == max_idletime][0]


class RenewalTheory(CSMethods):
    def __init__(self, name, db_path):
        CSMethods.__init__(self, name)
        self.idletimes = []
        self.db_path = db_path

    def update_idletimes(self, channel_state):
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        try:
            dataset = cursor.execute(
                f"""select * from {self.name}""").fetchall()
        except sqlite3.OperationalError:
            return
        if not len(dataset) > 0:
            return
        dataset = [[bit for i, bit in enumerate(row) if type(
            bit) is not str and i != 0] for row in dataset]
        mean_ONOFF = [[np.mean(times) if len(times) > 0 else 0.0 for times in channel] for channel in list(
            map(on_and_off_time, np.array(dataset).transpose()))]
        con.close()

        idletimes = []
        for i, (on, off) in enumerate(mean_ONOFF):
            lambda_on = np.ma.masked_values(on, 0)**-1
            lambda_off = np.ma.masked_values(off, 0)**-1
            if channel_state[i] == 0:
                prob = lambda_on * (lambda_on + lambda_off)**-1
                prob += (lambda_off * (lambda_on + lambda_off)**-1) * \
                    np.exp(-1*(lambda_on + lambda_off))
            else:
                prob = lambda_on * (lambda_on + lambda_off)**-1
                prob -= (lambda_on * (lambda_on + lambda_off)**-1) * \
                    np.exp(-1*(lambda_on + lambda_off))
            idletimes.append(prob * np.ma.masked_values(off, 0)**-1)

        self.idletimes = [i if np.ma.is_masked(
            i) is False else 0.0 for i in idletimes]
        return True

    def select_channel(self, channel_state):
        if all(channel_state):
            return
        if not self.update_idletimes(channel_state):
            return
        max_idletime = np.max(
            [idletime for i, idletime in self.idletimes if channel_state[i] == 0])
        if max_idletime == 0:
            return
        return [i for i, idletime in enumerate(self.idletimes) if idletime == max_idletime][0]
