# Given an input wav file and an output prefix, store the melscale transform to a new binary file.

import pickle
from melscale import melscale
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Write out melscale transforms')
parser.add_argument('wav_file', type=str, help='wav file path')
parser.add_argument('output_prefix', type=str, help='prefix for output file')

def main():
    args = parser.parse_args()
    audio_file = args.wav_file
    output_prefix = args.output_prefix 
    S = melscale(audio_file)
    output_path = os.path.join(output_prefix, os.path.split(audio_file)[-1] + '.mel')
    with open(output_path, "wb") as f: 
      pickle.dump(S, f)
      print("1 file written to " + output_path)

if __name__ == "__main__":
    main()
