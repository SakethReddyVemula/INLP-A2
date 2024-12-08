import argparse
import subprocess
import sys

def run_ffn_pos_tagger():
    subprocess.run(["python3", "ffn_pos_tagger.py"])

def run_lstm_pos_tagger():
    subprocess.run(["python3", "lstm_pos_tagger.py"])

def main():
    parser = argparse.ArgumentParser(description="Part-of-Speech Tagger Runner")
    parser.add_argument("-f", "--feedforward", action="store_true", help="Run feedforward neural network POS tagger")
    parser.add_argument("-r", "--recurrent", action="store_true", help="Run LSTM-based POS tagger")
    args = parser.parse_args()

    if args.feedforward:
        run_ffn_pos_tagger()
    elif args.recurrent:
        run_lstm_pos_tagger()
    else:
        print("Please specify either -f/--feedforward or -r/--recurrent to choose which POS tagger to run.")
        sys.exit(1)

if __name__ == "__main__":
    main()