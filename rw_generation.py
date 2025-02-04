import numpy as np
import matplotlib.pyplot as plt
import typer
from gluonts.dataset.arrow import ArrowWriter
import os

app = typer.Typer(pretty_exceptions_enable=False)


# -------- Geometric Brownian motion generation ---------
def generate_data(num_series, length_series, years, min_initial_price, max_initial_price, min_mu, max_mu, min_sigma, max_sigma):
    # initialize the step length
    dt = years/length_series
    # drift coefficent
    mu = np.random.uniform(min_mu, max_mu, size=(num_series,))
    # initial stock price
    initial_prices = np.random.uniform(min_initial_price, max_initial_price, size=(num_series,))
    # volatility
    sigma = np.random.uniform(min_sigma, max_sigma, size=(num_series,))
    # array of random noise
    normal_data = np.random.normal(0, np.sqrt(dt), size=(length_series, num_series))
    # generate all the requested series
    series = np.exp((mu-sigma**2/2)*dt + sigma * normal_data)
    # add one row to insert initial pries(the multiplier must start from one)
    series = np.vstack([np.ones(num_series), series])
    # multiply all columns, so that the series will be on the columns
    series = series.cumprod(axis = 0)
    # multiply the array by the initial prices, so that we obtain the final series
    series = initial_prices*series
    # return the transposed series, so that the rows are distinct series
    return series.T


def save_series(series, output_dir, filename='brownian_motions.arrow'):
    series_to_save = []
    for ts in series:
        series_to_save.append({"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts})
    os.makedirs(output_dir, exist_ok=True)
    ArrowWriter(compression="lz4").write_to_file(series_to_save, path=output_dir+filename)

    
@app.command()
def main(num_series:int= 1000,
         length_series:int=1024,
         years:float=0.5,
         min_initial_price:float=0,
         max_initial_price:float=300,
         min_mu:float=-0.10,
         max_mu:float=0.10,
         min_sigma:float=0.2,
         max_sigma:float=1,
         output_dir:str = './output/',
         display_first:int = 0,
         save:bool=True):

    assert display_first < num_series

    series = generate_data(num_series=num_series,
                           length_series=length_series,
                           years=years,
                           min_initial_price=min_initial_price,
                           max_initial_price=max_initial_price,
                           min_mu=min_mu,
                           max_mu=max_mu,
                           min_sigma=min_sigma,
                           max_sigma=max_sigma)
    time = np.linspace(0,years, length_series+1)
    if display_first > 0:
        for i in range(display_first):
            plt.plot(time, series[i])
            plt.xlabel("Years $(t)$")
            plt.ylabel("Stock Price $(S_t)$")
        plt.show()

    if save:
        save_series(series, output_dir=output_dir)

if __name__ == '__main__':
    app()


