import os
import pandas as pd
import json

def load_logfiles(
        filepath
    ) -> pd.DataFrame:
    filelist = [f for f in os.listdir(filepath) if f.endswith('.json') and f.startswith('Logfile_')]

    df_list = []

    for file in filelist:
        with open(os.path.join(filepath, file), 'r') as f:
            data = json.load(f)
        data_df = pd.json_normalize(data)
        data_df = data_df[[col for col in data_df.columns if not col.split('.')[-1] == 'Header']]
        df_list.append(data_df)

    df = pd.concat(df_list, ignore_index=True)
    return df

def main():
    FOLDERPATH = r'data\bipolar-Ti-100W-5ubar\BayBE Campaign_003831445664\Logfiles'
    print(load_logfiles(FOLDERPATH))

if __name__ == "__main__":
    main()