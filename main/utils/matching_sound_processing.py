"""
MATCHING PURSUIT WITH TIME-FREQUENCY DICTIONARY
"""
from utils.matching_pursuit import MatchingPursuit
import numpy as np
import librosa as lbr
import matplotlib.pyplot as plt

plt.style.use("ggplot")

class MSP():
    
    def __init__(self, target_path: list, source_path: list, sr: int = 44100) -> None:

        self.target, _ = lbr.load(target_path, sr=sr)
        self.source, _ = lbr.load(source_path, sr=sr)
        self.sr = sr

        self.matching_pursuit = MatchingPursuit(target=self.target, source=self.source)

    
    def generate_atoms(self, mode: str, **kwargs) -> list:

        """
        generate atoms and time-frequency dictionary

        mode: str, fixed or variable (see decomposition object)

        kwargs:
            wlen: win length
            hopsize: hop length in percent ([hopsize > 0 , to ...], hop * wlen)
            wlenmin: min win length
            wlenmax: max win length
            hopsizemin, hopsizemax: int, int min and max hopsize lenghts in percent ([hopsize > 0 , to ...], hop * wlen)
            nwin: number of lengths generated randomly (mode: variable)
        """

        param = {
            "wlen": 2048,
            "hopsize": 0.5,
            "wlenmin": 512,
            "wlenmax": 4096,
            "hopsizemin": 0.25,
            "hopsizemax": 0.75,
            "nwin": 10
        }

        param = param|kwargs

        print("\nGenerate target atoms...")

        self.matching_pursuit.generate_target_atoms(mode=mode, **param)
        
        print("\nDone!\n")
        print("Generate dictionary...\n")

        self.matching_pursuit.generate_dictionary()

        print("\nDone!\n")

        return self.matching_pursuit.target_atoms, self.matching_pursuit.dictionary
    
    def matching(self, k: int, eps: float):

        """
        perform matching pursuit

        k: max number of atoms to be extract
        eps: max error
        """

        self.matching_pursuit.matching(k=k, eps=eps)

    
    def perform_rebuild(self) -> list[float]:

        """
        rebuild target from matching atoms

        return: 1D vector, reconstructed target
        """
        
        matching_atoms = self.matching_pursuit.matching_atoms
        
        length = self.matching_pursuit.target_decomposition.pickup_points[-1] + len(matching_atoms[-1])

        y = np.zeros(length, dtype=float)

        for i, f in enumerate(matching_atoms):
            n = len(f)
            win = np.hanning(n)
            hop = self.matching_pursuit.target_decomposition.pickup_points[i]
            y[hop:hop+n] += f * win
        
        return y
    
    def plot_results(self):

        """
        plot results
        """

        fig, ax = plt.subplots(3, 2, figsize=(10, 10))

        t = 1/self.sr

        time_target = [i/self.sr for i in range(len(self.target))]
        ax[0, 0].plot(time_target, self.target)
        ax[0, 0].set_title("TARGET WAVEFORM")
        ax[0, 0].set_xlabel("time s.")
        ax[0, 0].set_ylabel("amp")

        tmag = np.abs(np.fft.rfft(self.target))
        tfreq = np.fft.rfftfreq(len(self.target), d=t)
        ax[0, 1].plot(tfreq, tmag)
        ax[0, 1].set_title("TARGET SPECTRUM")
        ax[0, 1].set_xlabel("freq Hz")
        ax[0, 1].set_ylabel("mag")

        time_source = [i/self.sr for i in range(len(self.source))]
        ax[1, 0].plot(time_source, self.source)
        ax[1, 0].set_title("SOURCE WAVEFORM")
        ax[1, 0].set_xlabel("time s.")
        ax[1, 0].set_ylabel("amp")

        smag = np.abs(np.fft.rfft(self.source))
        sfreq = np.fft.rfftfreq(len(self.source), d=t)
        ax[1, 1].plot(sfreq, smag)
        ax[1, 1].set_title("SOURCE SPECTRUM")
        ax[1, 1].set_xlabel("freq Hz")
        ax[1, 1].set_ylabel("mag")

        match_signal = self.perform_rebuild()
        time_match_signal = [i/self.sr for i in range(len(match_signal))]
        ax[2, 0].plot(time_match_signal, match_signal)
        ax[2, 0].set_title("MATCHING WAVEFORM")
        ax[2, 0].set_xlabel("time s.")
        ax[2, 0].set_ylabel("amp")

        mmag = np.abs(np.fft.rfft(match_signal))
        mfreq = np.fft.rfftfreq(len(match_signal), d=t)
        ax[2, 1].plot(mfreq, mmag)
        ax[2, 1].set_title("MATCHING SPECTRUM")
        ax[2, 1].set_xlabel("freq Hz")
        ax[2, 1].set_ylabel("mag")

        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        plt.show()