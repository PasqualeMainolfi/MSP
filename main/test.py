from utils.matching_sound_processing import MSP


TARGET = "/Users/pm/CloudStation/Drive/ComputerMusicCoding/MatchingPursuit/audio_file/vox.wav"
SOURCE = "/Users/pm/CloudStation/Drive/ComputerMusicCoding/MatchingPursuit/audio_file/classical.wav"

SR = 44100


if __name__ == "__main__":

    mp = MSP(target_path=TARGET, source_path=SOURCE, sr=SR)

    # create atoms and time-freq dictionary
    mp.generate_atoms(mode="variable", wlenmin=1024, wlenmax=4096, hopsizemin=0.25, hopsizemax=1, nwin=15)

    # generaete matching atoms
    mp.matching(k=10, eps=1e-6)

    # rebuild target
    y = mp.perform_rebuild()

    mp.plot_results()