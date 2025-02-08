import pandas as pd
from rw_generation import save_series
import numpy as np
from tqdm.auto import tqdm
import typer
from sklearn.model_selection import train_test_split

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
def main(base_path:str = './data/',
         n_samples:int = 5000,
         min_length:int = 128,
         max_length:int = 1024,
         seed:int = 0):

    # seed for replicability
    np.random.seed(seed)

    # create basic variables from symbols csv file
    all_symbols = pd.read_csv(base_path + 'symbols_valid_meta.csv')
    etfs = list(all_symbols[all_symbols['ETF']=='Y']['Symbol'])
    stocks = list(all_symbols[all_symbols['ETF']=='N']['Symbol'])
    all_data_list = []

    # load etf time series from names
    print('Loading ETFs...')
    for etf in tqdm(etfs):
        filename = base_path+'etfs/'+etf+'.csv'
        tmp_list = list(pd.read_csv(filename)['Adj Close'])
        all_data_list.append(tmp_list)

    print('Loading stocks...')
    # load stock time series from names
    for stock in tqdm(stocks):
        # correct ~3 incoherent names
        if('$' in stock): stock = stock.replace('$', '-')
        if(stock[-2:]=='.V'): stock = stock[:-2]+'#'
        # load data
        filename = base_path+'stocks/'+stock+'.csv'
        tmp_list = list(pd.read_csv(filename)['Adj Close'])
        all_data_list.append(tmp_list)

    # eliminate some noisy time series
    all_data_list = filter(lambda x: max(x)<1e6, all_data_list) # 10 time series had corrupted data, with unreasonably high values
    all_data_list = list(all_data_list)

    # split in train and test data
    train, test = train_test_split(all_data_list, test_size=0.3, shuffle=True, random_state=seed)

    # perform mixup
    print('Performing mixup on training data...')
    data_mixup = mixup(train, n_samples, min_length, max_length)

    # save datasets
    save_series(all_data_list, base_path, filename='total_data')
    save_series(test, base_path, filename='test_finance')
    save_series(train, base_path, filename='train_finance')
    save_series(data_mixup, base_path, filename=f'mixup_data_train_{n_samples}')

if __name__ == '__main__':
    app()
