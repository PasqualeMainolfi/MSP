
"""
define a DECOMPOSE object
"""

import numpy as np
import sys

class Decompose():
    
    def __init__(self, x: list) -> None:
        self.x = x
        self.n = len(x)

        self.decomposition_mode = None
        self.frames = None
        self.frame_lengths = None
        self.pickup_points = None

    
    def decompose(self, mode: str, **kwargs):


        """
        decompose signal to a short-time representation

        mode: str, decomposition mode [fixed, variable] 
                fixed: with fixed window and overlap factor
                variable = with variable window and overlap factor
        
        kwargs:
            wlen: int, window lengths in samples (fixed mode)
            hopsize: int, hop length in percent (wlen * hop)
            wlenmin, wlenmax: int, min and max window length (variable mode)
            hopsizemin, hopsizemax: int, int min and max hopsize lenghts in percent ([hopsize > 0 , to ...], hop * wlen)
            nwin: int, number of window length (variable mode)
        """
        
        try:
            assert mode in ["fixed", "variable"]
        except:
            print("[ERROR IN DECOMPOSE OBJECT] decomposition mode can be fixed or variable")
            sys.exit(0)
        
        self.decomposition_mode = mode

        param = {
            "wlen": 1024,
            "hopsize": 0.5,
            "wlenmin": 1024,
            "wlenmax": 4096,
            "hopsizemin": 0.25,
            "hopsizemax": 0.75,
            "nwin": 10
        }

        win_lengths = np.random.randint(
            low=param["wlenmin"], 
            high=param["wlenmax"],
            size=param["nwin"]
            )

        param = param|kwargs

        try:
            assert param["wlen"] < self.n
            assert param["hopsize"] > 0
            assert 0 < param["wlenmin"] <= param["wlenmax"]
            assert param["wlenmax"] >= param["wlenmin"]
            assert 0 <= param["hopsizemin"] <= param["hopsizemax"]
            assert param["hopsizemax"] >= param["hopsizemin"]
            assert param["nwin"] > 0
        except:
            print("[ERROR IN DECOMPOSE OBJECT] something is wrong with **kwargs")
            sys.exit(0)


        frames = []
        pickup_marks = []
        frame_lengths = set()

        hop, endwin = 0, 0
        while True:

            if mode == "fixed":
                wlen = param["wlen"]
                hopsize = round(wlen * param["hopsize"])
            elif mode == "variable":
                wlen = np.random.choice(win_lengths)
                hopmin, hopmax = round(wlen * param["hopsizemin"]), round(wlen * param["hopsizemax"])
                hopsize = np.random.randint(hopmin, hopmax + 1)
            
            w = np.hanning(wlen)

            if hop > self.n - wlen:
                if hop >= self.n:
                    break
                else:
                    f = self.x[hop:]
                    if mode == "variable":
                        end_frame = f
                        wlen = len(end_frame)
                    elif mode == "fixed":
                        end_frame = np.zeros(wlen)
                        end_frame[:self.n-hop] = f
                    
                    w = np.hanning(wlen)
                    frame_lengths.add(wlen)
                    pickup_marks.append(hop)
                    frames.append(end_frame * w)
                    break


            pickup_marks.append(hop)
            frame_lengths.add(wlen)

            frame = self.x[hop:hop+wlen]
            frames.append(frame * w)

            hop += hopsize

        self.frame_lengths = frame_lengths
        self.pickup_points = pickup_marks
        self.frames = frames








