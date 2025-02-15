import pandas as pd
import numpy as np
import os
from GBM_generation import save_series
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import typer
from sklearn.model_selection import train_test_split
from gluonts.dataset.arrow import ArrowWriter

app = typer.Typer(pretty_exceptions_enable=False)


def mean_scale(ts):
    """Given a time series perform the mean scale, dividing the time series by 
    the mean of the values, excluding nan values"""
    # ensure that the ts is a numpy array
    if isinstance(ts, list):
        ts = np.array(ts)
    # compute the mean and return the scaled time series
    mean = np.nanmean(ts)
    mean = mean if not np.isnan(mean) else 1
    ts /= mean
    return ts


def mixup(series, size, min_length, max_length, threshold=0.7, seed=None):
    """Compute TSMixup on series, returning a number of samples indicate in size, defining the min and maximum
    length for each series. Each series is mean scaled both before and after mixup."""

    # filter data and ensure that there is enough that is above min_length (according to the threshold)
    mixup_data = filter(lambda x: len(x)>=min_length, series)
    mixup_data = list(mixup_data)
    assert(len(mixup_data)>threshold*len(series))

    # set seed, in case it was not set before
    if seed is not None:
        np.random.seed(seed)

    # set global variables
    n_series = len(series)
    final_mixup = []

    # perform mixup
    for i in tqdm(range(size)):

        # initial variables and weights (that must sum to one)
        generated = False
        n_mixup = np.random.randint(1,4)
        weights = np.random.random(size=n_mixup)
        weights /= sum(weights)

        # keep resampling until all the time series are longer than the required length
        while not generated:
            length = np.random.randint(min_length, max_length)
            indices = np.random.randint(0, len(mixup_data), size=n_mixup)
            bool_lengths = [len(mixup_data[idx])>length for idx in indices]
            generated = np.all(bool_lengths)

        # initialize ts
        mixed_ts = np.zeros(length)
        
        # generate the time series by weighting the three ts obtained
        for idx,weight in zip(indices, weights):
            tmp_series = mean_scale(mixup_data[idx]) # scale time series
            start_idx = np.random.randint(0, len(tmp_series)-length) # sample the starting index
            final_idx = start_idx+length # get the final index
            mixed_ts += tmp_series[start_idx:final_idx] * weight # weight the time series and add it to the current one
        # scale the final time series and append
        mixed_ts = mean_scale(mixed_ts)
        final_mixup.append(mixed_ts)

    return final_mixup


@app.command()
def main(base_path:str = '/content/data',
         out_path:str = '/content',
         mixup_samples:int = 45000,
         min_length:int = 128,
         max_length:int = 1024,
         seed:int = 42):

    files = list(filter(lambda x: x[-3:]=='csv', os.listdir(base_path)))
    train_files, test_files = train_test_split(files,test_size=0.3, shuffle=True, random_state = seed)

    print('Creating train files...')
    train = []
    for filename in tqdm(train_files):
        tmp_df = pd.read_csv(f"{base_path}/{filename}")
        start = pd.Timestamp(tmp_df['date'][0])
        target = np.array(tmp_df['close'])
        train.append({'start':start, 'target':target})


    print('Creating test files...')
    test = []
    for filename in tqdm(test_files):
        tmp_df = pd.read_csv(f"{base_path}/{filename}")
        final_idx = tmp_df['close'].shape[0]%10
        tmp_df_end = tmp_df.iloc[-final_idx:]
        list_dfs = np.split(tmp_df[final_idx:], 10)
        list_dfs[-1] = pd.concat([list_dfs[-1], tmp_df_end])
        for df in list_dfs:
            df = df.reset_index()
            start = pd.Timestamp(df['date'][0])
            target = np.array(df['close'])
            test.append({'start':start, 'target':target})


    train_filename = 'train_dataset_5min'
    test_filename = 'test_dataset_5min'
    ArrowWriter(compression="lz4").write_to_file(test, path=f"{out_path}/{test_filename}.arrow")
    ArrowWriter(compression="lz4").write_to_file(train, path=f"{out_path}/{train_filename}.arrow")

    series_for_mixup = [series['target'] for series in train]
    mixed_series = mixup(series_for_mixup,45000, min_length=128, max_length=1024, seed=42)
    mixup_filename = 'mixup_dataset'
    save_series(mixed_series, output_dir=out_path, filename=mixup_filename)

if __name__ == '__main__':
    app()
