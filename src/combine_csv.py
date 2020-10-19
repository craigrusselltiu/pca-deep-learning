import pandas as pd
import sys


# Read and filter csv files, then combined them into master csv
def combine(img, img_csv, find_csv):
    img_df = pd.read_csv(img_csv)
    img_df = img_df[img_df['Name'].str.contains(img)]

    find_df = pd.read_csv(find_csv)

    frames = [img_df.reset_index(drop=True), find_df.reset_index(drop=True)] # reset indices
    combined = pd.concat(frames, axis = 1) # concatenate horizontally
    combined = combined.loc[:, ~combined.columns.duplicated()] # drop duplicate columns
    combined.to_csv('../lib/' + img + '.csv')


def main():
    img_csv = '../lib/ProstateX-2-Images-Train.csv'
    find_csv = '../lib/ProstateX-2-Findings-Train.csv'

    combine(sys.argv[1], img_csv, find_csv)


if __name__ == '__main__':
    main()