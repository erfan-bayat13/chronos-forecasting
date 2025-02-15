import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import typer


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(data_path = '/content/drive/MyDrive/Financial Extension/New Dataset/Results',
         out_path = '/content'):
    
    # initialize lists
    mase_gbm = []
    mase_no_ft = []
    wql_gbm = []
    wql_no_ft = []

    # read files and fill lists
    for i in range(6,96, 6):
        df_gbm = pd.read_csv(f"{data_path}/results_after_finetuning_10_perc_{i}_pl.csv")
        df_no_ft = pd.read_csv(f"{data_path}/results_before_finetuning_{i}_pl.csv")
        mase_gbm.append(df_gbm['MASE'].values[0])
        mase_no_ft.append(df_no_ft['MASE'].values[0])
        wql_gbm.append(df_gbm['WQL'].values[0])
        wql_no_ft.append(df_no_ft['WQL'].values[0])

    # trasform lists in numpy arrays (useful at the end to compute the percentages)
    mase_gbm = np.array(mase_gbm)
    mase_no_ft = np.array(mase_no_ft)
    wql_gbm = np.array(wql_gbm)
    wql_no_ft = np.array(wql_no_ft)

    # plot the MASE
    plt.plot(range(6,96, 6), mase_gbm, label='Finetuned model', marker='.')
    plt.plot(range(6,96, 6), mase_no_ft, label='Original model',  marker='.')
    plt.xlabel("Prediction length")
    plt.ylabel("MASE")
    plt.title("MASE of finetuned model vs. original")
    plt.legend()
    plt.savefig(f"{out_path}/mase_finetuned_vs_original.png")
    plt.show()

    # plot the WQL
    plt.plot(range(6,96, 6), wql_gbm, label='Finetuned model', marker='.')
    plt.plot(range(6,96, 6), wql_no_ft, label='Original model', marker='.')
    plt.xlabel("Prediction length")
    plt.ylabel("WQL")
    plt.title("WQL of finetuned model vs. original")
    plt.legend()
    plt.savefig(f"{out_path}/wql_finetuned_vs_original.png")
    plt.show()


    print("Percentage performance increase (MASE): ",  (mase_no_ft -mase_gbm)/mase_no_ft)
    print("Percentage performance increase (WQL): ", (wql_no_ft -wql_gbm)/wql_no_ft)

if __name__ == '__main__':
    app()

