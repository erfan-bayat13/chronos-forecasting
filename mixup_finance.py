import pandas as pd
from rw_generation import save_series
import numpy as np
from tqdm.auto import tqdm
import typer

app = typer.Typer(pretty_exceptions_enable=False)

def mixup(series, size, length, threshold=0.2):
    mixup_data = filter(lambda x: len(x)>=1024, series)
    mixup_data = list(mixup_data)
    assert(len(mixup_data)>threshold*len(series))
    n_series = len(series)
    final_mixup = np.zeros((size, length))
    for i in tqdm(range(size)):
        n_mixup = np.random.randint(1,4)
        indices = np.random.randint(0, len(mixup_data), size=n_mixup)
        weights = np.random.random(size=n_mixup)
        weights /= sum(weights)
        for idx,weight in zip(indices, weights):
            tmp_series = np.array(mixup_data[idx])
            start_idx = np.random.randint(0, len(tmp_series)-length+1)
            final_idx = start_idx+length
            final_mixup[i] += tmp_series[start_idx:final_idx]*weight
    return final_mixup

@app.command()
def main(base_path:str = './data/',
         n_samples:int = 5000,
         length_samples:int = 1024):
    all_symbols = pd.read_csv(base_path + 'symbols_valid_meta.csv')
    etfs = list(all_symbols[all_symbols['ETF']=='Y']['Symbol'])
    stocks = list(all_symbols[all_symbols['ETF']=='N']['Symbol'])
    all_data_list = []
    print('Loading ETFs...')
    for etf in tqdm(etfs):
        filename = base_path+'etfs/'+etf+'.csv'
        tmp_list = list(pd.read_csv(filename)['Adj Close'])
        all_data_list.append(tmp_list)
    print('Loading stocks...')
    for stock in tqdm(stocks):
        if('$' in stock): stock = stock.replace('$', '-')
        if(stock[-2:]=='.V'): stock = stock[:-2]+'#'
        filename = base_path+'stocks/'+stock+'.csv'
        tmp_list = list(pd.read_csv(filename)['Adj Close'])
        all_data_list.append(tmp_list)

    save_series(all_data_list, base_path, filename='original_data.arrow')
    data_mixup = mixup(all_data_list, n_samples, length_samples)
    save_series(data_mixup, base_path, filename=f'mixup_data_{n_samples}.arrow')

if __name__ == '__main__':
    app()
