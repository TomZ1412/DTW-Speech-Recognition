from utils import vad_energy_tcr, dtw_distance, extract_features
import numpy as np
import librosa
import pickle
import os
import argparse

def recognize_digit(signal, fs, templates,mode):
    start, end = vad_energy_tcr(signal, fs, mode,plot=True)
    best_digit_list = []
    # print((end-start)/fs)
    for i in range(len(start)):
        best_digit = None
        seg = signal[start[i]:end[i]]
        feat_lpc, feat_fft = extract_features(seg, fs)
        # print(feat_lpc.shape, feat_fft.shape)
        # feat= extract_features(seg, fs)
        # print(feat_fft.shape)
        # print(feat_fft[:,0])
    # import pdb; pdb.set_trace()
        min_dist = float('inf')
        for digit in range(10):
            dist_lpc = dtw_distance(feat_lpc, templates[digit][0])/feat_lpc.shape[0]
            dist_fft = dtw_distance(feat_fft, templates[digit][1])/feat_fft.shape[0]
            dist = dist_lpc + dist_fft
            # print(dist_lpc, dist_fft)
            # dist = dist_fft
            # dist = dtw_distance(feat_lpc, templates[digit][0])
            # dist = dtw_distance(feat, templates[digit])
            # print(f"Digit {digit}: Distance = {dist:.2f}") 
            if dist <= min_dist:
                min_dist = dist
                best_digit = digit
        best_digit_list.append(best_digit)
    return best_digit_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--mode", type=str, default="single")
    args = parser.parse_args()
    
    test_dir = args.test_dir
    mode = args.mode
    templates= pickle.load(open("templates.pkl", "rb"))
    acc=0
    acc_multi = 0
    if mode == "single":
        for filename in os.listdir(test_dir):
            if filename.endswith("16000hz.wav"):
                signal, fs = librosa.load(os.path.join(test_dir, filename), sr=None)
                print(f"Processing {filename}...")
                pred = recognize_digit(signal, fs, templates,mode)
                print(f"Predicted digit: {pred}")
                # print(recognize_digit(signal, fs, templates))
                # if filename.startswith(str(pred)):
                if pred[0] == int(filename.split("_")[0]):
                    acc += 1
    elif mode == "multi":
        for filename in os.listdir(test_dir):
            if filename.endswith("16000hz.wav"):
                signal, fs = librosa.load(os.path.join(test_dir, filename), sr=None)
                print(f"Processing {filename}...")
                pred = recognize_digit(signal, fs, templates,mode)
                print(f"Predicted digit: {pred}")
                correct = True
                # if pred == [2,8]:
                #     import pdb; pdb.set_trace()
                for i in range(len(pred)):
                    if pred[i] != int(filename[i]):
                        correct = False
                    else:
                        acc += 1
                if correct:
                    acc_multi += 1
    DATA_LENGTH=10 # CHANGE THIS TO YOUR TEST DATA LENGTH
    acc /= DATA_LENGTH
    # acc_multi /= 6
    print(f"Accuracy: {acc:.2%}")
    # print(f"Accuracy_multi: {acc_multi:.2%}")
            
