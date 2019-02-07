import pickle
from utils import melspectrogram
import numpy as np

def main():
    #Load all data and labels
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)

    train_set_data = train_set['data']
    train_set_labels = train_set['labels']
    train_set_id = train_set['track_id']

    test_set_data = test_set['data']
    test_set_labels = test_set['labels']
    test_set_id = test_set['track_id']


    
    train_mel = []
    #Convert audio data to spectrogram
    for i in range(np.shape(train_set_data)[0]):
        train_mel.append(melspectrogram(train_set_data[i][:]))
        print(i/11250)
    test_mel = []
    for i in range(np.shape(test_set_data)[0]):
        test_mel.append(melspectrogram(test_set_data[i][:]))
        print(i/3750)

    return train_mel,test_mel,train_set_labels,test_set_labels,test_set_id
    


if __name__ == '__main__':
    main()