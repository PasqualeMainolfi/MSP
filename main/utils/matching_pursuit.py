"""
Matching Pursuit with Time-Frequency Dictionary

source: Matching Pursuit with Time-Frequency Dictionary, S. G. Mallat, Z. Zhang, IEEE transaction 1993
"""

from utils.decompose import Decompose
import numpy as np
from tqdm import tqdm

def L2(x: list):
    return np.sqrt(np.sum(np.square(x)))

def avoid_zero(x: list):
    y = np.copysign(x, 0)
    y = x + 1e-12 if not np.any(y) else x
    return y


class MatchingPursuit():
    def __init__(self, target, source) -> None:
        self.target = target
        self.source = source

        self.target_decomposition = Decompose(x=self.target) # target decomposition 
        self.source_decomposition = Decompose(x=self.source) # source decomposition

        self.target_atoms = None
        self.dictionary = None

        self.coeffs = None # coeffs with best index from matching pursuit process
        self.atoms = None # atoms with best index from matching pursuit process

        self.matching_atoms = None # matrix result of the product between coeffs and atoms with best index during the process
        
    
    def generate_target_atoms(self, mode: str, **kwargs) -> list[list[float]]:

        """
        generates target atoms

        mode: str, fixed or variable (see decomposition object)

        kwargs:
            wlen: win length
            hopsize: hop length in percent ([hopsize > 0 , to ...], hop * wlen)
            wlenmin: min win length
            wlenmax: max win length
            hopsizemin, hopsizemax: int, int min and max hopsize lenghts in percent ([hopsize > 0 , to ...], hop * wlen)
            nwin: number of lengths generated randomly (mode: variable)

        return: list[list[float]]
        """

        param = {
            "wlen": 1024,
            "hopsize": 0.5,
            "wlenmin": 1024,
            "wlenmax": 4096,
            "hopsizemin": 0.25,
            "hopsizemax": 0.75,
            "nwin": 10
        }

        param = param|kwargs

        self.target_decomposition.decompose(mode=mode, **param) # decompose target

        frames = self.target_decomposition.frames

        target_atoms = []
        for i in tqdm(range(len(frames))):
            f = frames[i]
            fft = np.fft.rfft(f).real
            target_atoms.append(fft)
        
        tp = float if mode == "fixed" else object
        target_atoms = np.array(target_atoms, dtype=tp)
        self.target_atoms = target_atoms
    
    def generate_dictionary(self) -> dict():

        """
        generate time-frequency dictinary
        """
        
        frame_lengths = self.target_decomposition.frame_lengths

        dictionary = dict()
        for i in tqdm(range(len(frame_lengths))):
            length = list(frame_lengths)[i]
            self.source_decomposition.decompose(mode="fixed", wlen=length, hopsize=0.5)
            g = np.hamming(length)
            temp_frames = []
            for frame in self.source_decomposition.frames:
                frame = avoid_zero(x=frame)
                fft = np.fft.rfft(frame * g).real
                fft = fft/np.linalg.norm(fft) # norm = 1
                temp_frames.append(fft)
            key = length//2 + 1
            dictionary[key] = np.array(temp_frames)
        
        self.dictionary = dictionary

    
    def find_coeffs_and_atoms(self, x: list, dictionary: list, k: int, eps: float):

        """
        k: max number of atoms to be extract
        eps: max error
        """
        
        x = avoid_zero(x=x)

        r = x.copy() # residual
        d = dictionary.copy() # dictionary
        coeffs, atoms = [], []

        l2_sig = L2(x=x)

        i = 0
        while L2(x=r)/l2_sig > eps:

            dot = np.dot(d, r)
            max_ndx = np.argmax(np.abs(dot))

            coeffs.append(dot[max_ndx])
            atoms.append(d[max_ndx])

            d = np.delete(arr=d, obj=max_ndx, axis=0) # remove duplicate

            r = r - coeffs[-1] * atoms[-1]

            i += 1
            if i == k or len(d) == 0:
                break
        
        self.coeffs = np.array(coeffs)
        self.atoms = np.array(atoms)
    
    def matching(self, k: int, eps: float) -> list:

        """
        generate matching atoms

        k: max number of atoms to be extract
        eps: max error

        return: list[float], matching vector
        """

        print("\nMATCHING ATOMS...\n")

        m = []
        for i in tqdm(range(len(self.target_atoms))):
            frame = self.target_atoms[i]
            n = len(frame)
            self.find_coeffs_and_atoms(x=frame, dictionary=self.dictionary[n], k=k, eps=eps)
            atom = np.sum(self.coeffs * self.atoms.T, axis=1)
            atom_ifft = np.fft.irfft(atom)
            m.append(atom_ifft)
        
        print("\nDONE!\n")

        tp = float if self.target_decomposition.decomposition_mode == "fixed" else object
        m = np.array(m, dtype=tp)
        self.matching_atoms = m
        


