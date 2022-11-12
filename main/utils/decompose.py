
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

        mode: str, decomposition mode [static, dynamic] 
                static: with fixed window and overlap factor
                dynamic = with variable window and overlap factor
        
        kwargs:
            wlen: int, window lengths in samples (static mode)
            hopsize: int, hop length in percent (wlen * hop)
            wlenmin, wlenmax: int, min and max window length (dynamic mode)
            hopsizemin, hopsizemax: int, int min and max hopsize lenghts in percent ([hopsize > 0 , to ...], hop * wlen)
            nwin: int, number of window length (dynamic mode)
        """
        
        try:
            assert mode in ["static", "dynamic"]
        except:
            print("[ERROR IN DECOMPOSE OBJECT] decomposition mode can be static or dynamic")
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

        hop = 0
        while True:

            if mode == "static":
                wlen = param["wlen"]
                hopsize = round(wlen * param["hopsize"])
            elif mode == "dynamic":
                wlen = np.random.choice(win_lengths)
                hopmin, hopmax = round(wlen * param["hopsizemin"]), round(wlen * param["hopsizemax"])
                hopsize = np.random.randint(hopmin, hopmax + 1)
            
            w = np.hanning(wlen)
            pickup_marks.append(hop)
            frame_lengths.add(wlen)

            if self.n - hop < wlen:
                break

            frame = self.x[hop:hop+wlen]
            frames.append(frame * w)

            hop += hopsize
        
        end_frame = np.zeros(wlen)
        end_index = self.n - hop
        end_frame[:end_index] = self.x[hop:]
        frames.append(end_frame * w)

        self.frame_lengths = frame_lengths
        self.pickup_points = pickup_marks
        self.frames = frames








