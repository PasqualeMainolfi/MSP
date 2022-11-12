from utils.mp import MP
import librosa as lbr


TARGET = "/Users/pm/CloudStation/Drive/ComputerMusicCoding/MatchingPursuit/audio_file/vox.wav"
SOURCE = "/Users/pm/CloudStation/Drive/ComputerMusicCoding/MatchingPursuit/audio_file/classical.wav"
SR = 44100
WLEN = 4096
HOPSIZE = 1024


if __name__ == "__main__":

    mp = MP(target_path=TARGET, source_path=SOURCE)

    # create atoms and time-freq dictionary
    mp.generate_atoms(mode="dynamic", wlenmin=1024, wlenmax=16387, hopsizemin=0.25, hopsizemax=0.75, n_win=10)

    # generaete matching signal
    mp.matching(k=10, eps=1e-6)

    # rebuild target
    y = mp.perform_rebuild()

    mp.plot_results()