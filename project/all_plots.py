import matplotlib.pyplot as plt
from librosa.display import specshow

#'''
#metal_pop_spectro
def main(*spectro):
    for idx,i in enumerate(spectro):
        plt.subplot(1, 2, idx+1)
        specshow(i)
        plt.ylabel('Frequency (kHz)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
    plt.suptitle('Figure 1: Spectrograms of two tracks of the GTZAN dataset.\
    \nThe left one represents a pop track while the second one represents a metal track.',fontsize=25)

    plt.show()
if __name__ == '__main__':
    main()
#'''