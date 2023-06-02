import re
import sys
import argparse
from tqdm import tqdm

def avg_repetitions(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    repetitions = []

    for line in tqdm(lines):
        if line.startswith('D-'):
            sentence = re.split('\t', line)[-1].strip()
            words = sentence.split()
            word_counts = {}

            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            
            repetition_count = sum([count-1 for count in word_counts.values() if count > 1])
            repetitions.append(repetition_count)
    
    return sum(repetitions) / len(repetitions) if repetitions else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='File to count repetitions in')
    args = parser.parse_args()

    print(avg_repetitions(args.filename))
