import random
import tflite_runtime.interpreter as tflite
import os
import numpy as np
from .utils import start_and_idle_time


class RandomChannelSelection:
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
        self.selected_channel = None
        self.name = "random"

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

        free_channels = []
        for i, state in enumerate(channel_state):
            if int(state) == 0:
                free_channels.append(i)
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


class NextOrPreviousChannelSelection:
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
        self.selected_channel = None
        if state not in ["next", "prev"]:
            raise Exception("State can only be next or previous")
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


class ML:
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

    def __init__(self, name: str, window_size: int = 10, model_path: str = os.path.abspath("./models/model.tflite")):
        self.dataset: list = []
        self.name = name
        self.window_size: int = window_size
        self.model_interpreter = tflite.Interpreter(model_path=model_path)
        # allocate memory
        self.model_interpreter.allocate_tensors()
        self.selected_channel = None

    def update_dataset(self, channel_state: list) -> None:
        """Update the dataset buffer

           Parameters
           ----------
                channel_state: (1-D Array-like) state of the channel

            Return 
            ------
                None
        """
        self.dataset.append(channel_state)

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
        self.update_dataset(channel_state)

        if not len(self.dataset) >= self.window_size:
            return

        self.dataset = self.dataset[:self.window_size]

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

    def select_channel(self, channel_state):
        pass


class CS1(ML):
    """Channel Selection Algorithm 1

       Make inference on the next state of the channel and selects a channel 
       with the least occupancy

       Attributes
       ----------
        occ_sum: (list)
        occupancies: (list)
        counter: (int)
    """

    def __init__(self, name: str, nchannels: int, window_size: int = 10, model_path=...):
        super().__init__(name, window_size, model_path)
        self.occ_sums = []
        self.occupancies = [0, ] * nchannels
        self.counter = 0

    def update_dataset(self, channel_state):
        self.dataset.append(channel_state)
        self.occupancies = self.update_occupancies(channel_state)

    def update_occupancies(self, channel_state):
        """Compute the latest occupancies of the channel

           Steps:
                update the sums of states and increment counter
                new occupancy = sum of states / counter

            Parameters
            ----------

        """
        self.occ_sums = [r+channel_state[i]
                         for i, r in enumerate(self.occ_sums)]
        self.counter += 1
        return [o / self.counter for o in self.occ_sums]

    def select_channel(self, channel_state):
        """Make predictions and select the channel with the least occupancy

           Parameters
           ----------
                channel_state: (1-D array-like) state of the channel

            Returns
            -------
                None or selected_channel
        """
        channel_state = list(channel_state)
        # get predictions
        preds = self.get_prediction(channel_state)
        if preds is None:
            return

        preds = (np.array(preds).flatten() > 0.5).astype(int)
        free_channels = [i for i, s in enumerate(preds) if s == 0]
        # no channel is free
        if len(free_channels) == 0:
            return
        if self.selected_channel in free_channels:
            return self.selected_channel
        # only one channel
        if len(free_channels) == 1:
            self.selected_channel = free_channels[0]
            return self.selected_channel
        # select the least occupancy
        latest_occ = [i for i, _ in enumerate(
            self.occupancies) if i in free_channels]
        self.selected_channel = [i for i, val in enumerate(
            latest_occ) if val == min(latest_occ)][0]
        return self.selected_channel


class CS2(ML):
    """Channel Selection Algorithm 2

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
            update_dataset(channel_state)

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
        self.update_dataset(channel_state)
        if not len(self.dataset) >= self.window_size:
            return
        # prepare the data
        self.prepare_dataset()
        # get predictions
        preds = self.get_prediction(channel_state)
        preds = np.array(preds).flatten()

        if preds is None:
            return
        # if no channel is free
        if max(preds) <= 0:
            return
        # select channel of highest idle time
        self.selected_channel = [
            i for i, val in enumerate(preds) if val == max(preds)][0]
        return self.selected_channel
