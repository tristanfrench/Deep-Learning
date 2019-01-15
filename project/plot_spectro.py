import matplotlib.pyplot as plt
from librosa.display import specshow

def main(*spectro):
    for idx,i in enumerate(spectro):
        plt.subplot(1, 2, idx+1)
        specshow(i)
    plt.show()
if __name__ == '__main__':
    main()