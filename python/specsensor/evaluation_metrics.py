import time

class MeasureSwitchRate:
    """A evaluation metric to measure the effectiveness of a channel selection algorithm
       Attributes
       ----------
            selected_channel: (int) channel actively selected
            swtich_counts: (int) counter for the number of switches done by the algorithm
            cs_method: (string) the channel selection method in use
            start_time: (int) time at the start of episode
            result: (float) the computed switch rate at the end of the episode

       Methods
       -------
            count_switch_rate(selected_chan: int) method that compute the latest switch rate 
            given the selected channel of the algorithm

    """
    def __init__(self, selected_chan: int):
        self.selected_channel = selected_chan
        self.switch_counts = 0
        self.cs_method = ""
        self.start_time = time.time()
        self.result = self.get_switch_rate()

    def count_switch_rate(self, selected_chan: int) -> None:
        """Increment counts if selected channel changes and updates switch rate
           
           Parameters
           ----------
                selected_chan: (int) selected channel by the algorithm

           Return
           ------
                None
        """
        if selected_chan == None:
            return
        if selected_chan != self.selected_channel:
            self.switch_counts += 1
        self.selected_channel = selected_chan
        self.result = self.get_switch_rate()

    def get_switch_rate(self):
        """Compute the switch rate
           new switch rate =  number of switches / time elasped

           Parameters
           ----------
           None

           Return
           ------
                switch_counts / time_elasped: (float)
        """
        return self.switch_counts / \
            (time.time() - self.start_time)

    def get_switch_count(self):
        return self.switch_counts

    def set_cs_method(self, cs_method):
        self.cs_method = cs_method

    def get_cs_method(self):
        return self.cs_method
