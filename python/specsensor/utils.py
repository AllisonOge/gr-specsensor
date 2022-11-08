import numpy as np

# utility function
def start_and_idle_time(bit_sequence):
    bit_sequence = np.asarray(bit_sequence, dtype=np.int32)
    start_time = 0
    idle_time = 0
    idle_times = []
    for i, bit in enumerate(bit_sequence):
        # print(i, bit)
        if i > 0:
            if bit == 0 and bit_sequence[i-1] == 1:
                # 1,0
                start_time = i
                idle_time += 1
            elif bit == 0 and bit_sequence[i-1] == 0:
                # 0,0
                idle_time += 1
            elif bit == 1 and bit_sequence[i-1] == 0:
                # 0,1
                idle_times.append((start_time, idle_time))
                idle_time = 0
            else:
                # 1,1
                continue
        else:
            if bit == 0:
                start_time = i
                idle_time += 1

        if i == len(bit_sequence)-1 and idle_time > 0:
            idle_times.append((start_time, idle_time))

    return np.array(idle_times)

def on_and_off_time(bitseq):
    bitseq = np.asarray(bitseq, dtype=np.int32)
    on_time = 0
    off_time = 0
    on_times = []
    off_times = []
    for i, bit in enumerate(bitseq):
        # print(i, bit)
        if i > 0:
            if bit == 0 and bitseq[i-1] == 1:
                # 1,0
                on_times.append(on_time)
                on_time = 0
                off_time += 1
            elif bit == 0 and bitseq[i-1] == 0:
                # 0,0
                off_time += 1
            elif bit == 1 and bitseq[i-1] == 0:
                # 0,1
                off_times.append(off_time)
                off_time = 0
                on_time += 1
            else:
                # 1,1
                on_time += 1
        else:
            if bit == 0:
                off_time += 1
            else:
                on_time += 1

        if i == len(bitseq)-1 and off_time > 0:
            off_times.append(off_time)
        if i == len(bitseq)-1 and on_time > 0:
            on_times.append(on_time)

    return np.array(on_times), np.array(off_times)
