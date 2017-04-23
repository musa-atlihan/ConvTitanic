import pandas as pd
import numpy as np



def uniq2vec(series):
    """
    Take each column and assign a unique vector for each unique string.
    """
    vals = series.fillna('no_val').unique()
    ls = []
    size = vals.size
    maxLen = int(len(bin(size-1)[2:]))
    i=0
    for val in np.nditer(vals, ["refs_ok"]):
        binar = bin(i)[2:]
        while len(binar) != maxLen:
            binar = '0'+binar
        ls.append([val, binar])
        i += 1
    #np.savetxt('someFileName', ls, fmt='%s');
    return ls


def vec2list(series, binList):
    """
    Create lists with vector representations for those string values on each column.
    """
    fullList = series.fillna('no_val').tolist()
    ls = []
    for x in xrange(len(fullList)):
        for y in xrange(len(binList)):
            if fullList[x] == binList[y][0]:
                ls.append(binList[y][1])
    return pd.DataFrame(ls)


def vec2multi(df):
    """
    Separate elements of each vector to multiple columns.
    """
    mList = df[0].tolist()
    ls = []
    for x in xrange(len(mList)):
        ls.append(list(mList[x]))
    df = pd.DataFrame(ls)
    return df


def uniq2multi(series, col_name=None):
    ls = uniq2vec(series)
    df = vec2list(series, ls)
    mdf = vec2multi(df)
    if col_name is None:
        col_name = series.name
    mdf.columns = [str(col_name) + '_' + str(col) for col in mdf.columns]
    return mdf


def nan2mean(series):
    """
    Take nans on a pandas.series object and replace with 
    the mean of existing values.

    """
    ls = []
    mean = np.nanmean(series)
    for val in np.nditer(series, ["refs_ok"]):
        if val != val:
            ls.append(float(mean))
        else:
            ls.append(float(val))
    df = pd.DataFrame(ls)
    df.columns = [str(series.name)]
    return df


def char2num(series, char_mapping):
    """
    Take characters on each row on a pandas.series object and
    replace with an integer with a given char_mapping.

    """
    max_len = series.str.len().max()
    arr = np.zeros((len(series), int(max_len)))
    series = series.replace(np.nan, ' ', regex=True)
    for i, item in enumerate(series):
        ls = [char_mapping[str(char).lower()] for char in list(item)]
        if len(ls) < max_len:
            ls.extend([0] * (int(max_len) - len(ls)))
        arr[i,:] = ls

    df = pd.DataFrame(arr)
    df.columns = [str(series.name) + '_' + str(col) for col in df.columns]
    return df



def char2quan(series_list, char_mapping, width, height):
    """
    character quantization:

    Take characters on each row on a pandas.series object and
    create vector representations for each character as the width and
    word representation as the height of a matrix(image).
    """
    arr = np.zeros((len(series_list[0]), len(series_list), height, width), dtype=np.int32)
    for k, series in enumerate(series_list):
        series = series.replace(np.nan, ' ', regex=True)
        for i, row in enumerate(series):
            for j, char in enumerate(list(row)):
                if j < height:
                    arr[i, k, j, char_mapping[str(char).lower()]] = 1
    return arr