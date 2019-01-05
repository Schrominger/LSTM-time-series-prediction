import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

import os


def get_vwap_volume_product(vwap_df, volume_df):
    '''
     Calculate the full data VWAP values
    Input: two DataFrame with timestamp
    '''
    vwap_all =  (vwap_df*volume_df)

    return vwap_all

def get_days_sum_data(df, day):

    return df[str(pd.Period(day, freq = 'D'))].sum()


def get_vwap_for_days(start, end, vwap_all, volume_df):
    '''
    Calculate vwap value in timespan from start to end.
    -----
    start, end: timestamp or datetime objects
    return: vwap of all products for a single day if end=None
    '''
    if end is None:
        # calculate the whole day vwap summation
        return np.array(
        vwap_all[str(pd.Period(start, freq = 'D'))].sum()
        /volume_df[str(pd.Period(start, freq = 'D'))].sum()
        )
    else:
        # given time span vwap calculation
        return np.array(vwap_all[str(start):str(end)].sum()/volume_df[str(start):str(end)].sum())

def combineTimestamp(date, hourMin):
    """ Calculate the timestamp from given tradeDate, hourMinute file"""
    # match hour and minute
    # note that minute data has 60 which is large than 59
    str_hM = str(hourMin).split('.')[0]
    str_hM = (4-len(str_hM))*'0' + str_hM

    timeStamp = datetime.strptime(str(date), '%Y%m%d.0')+pd.to_timedelta(float(str_hM[0:2])*3600+float(str_hM[2:])*60, unit='s')

    return timeStamp

def cleanData():
    Date = pd.read_csv('./data/tradeDate5min_.csv', index_col=0)
    hourMinute = pd.read_csv('./data/hourMinute_.csv', index_col=0)
    prcVwap = pd.read_csv('./data/prcVwap_.csv', index_col=0)
    volume = pd.read_csv('./data/volume_.csv', index_col=0)
    # 将时间戳合并生成TimeIndex
    timeStampList = [combineTimestamp(x,y) for x,y in zip(Date['0'].values, hourMinute['0'].values)]
    timeIndex = pd.DatetimeIndex(timeStampList)

    # 重新定义index为时间戳
    prcVwap_timeIndex = prcVwap.rename(index = dict(zip(prcVwap.index, timeIndex)))
    volume_timeIndex = volume.rename(index = dict(zip(volume.index, timeIndex)))

    # save data
    if not os.path.exists('./newdata'):
        os.makedirs('./newdata')
        prcVwap_timeIndex.to_csv('./newdata/timeIndex_prcVwap_.csv')
        volume_timeIndex.to_csv('./newdata/timeIndex_volume_.csv')
        print('Save to csv !')

def split_df_into_goods_with_return():
    '''
    Split the DataFrame into small csv for each goods!

    '''
    path = './newdata/goods'
    if not os.path.exists(path):
        os.makedirs(path)
        if os.path.exists('./newdata/vwap_for_days_.csv') and os.path.exists('./newdata/volume_for_days_.csv'):
            vwap_for_days = pd.read_csv('./newdata/vwap_for_days_.csv', index_col=0, parse_dates = True)
            volume_days = pd.read_csv('./newdata/volume_for_days_.csv', index_col=0, parse_dates = True)

            for id in range(37):
                # note the fillna assumption
                df_vwap = pd.DataFrame(vwap_for_days[str(id)]).fillna(method = 'ffill').fillna(method = 'bfill')
                df_vol = pd.DataFrame(volume_days[str(id)]).fillna(0.0) # fill zeros for volume

                goods_df = pd.concat([df_vwap, df_vol], axis=1)#.dropna()
                goods_df.columns = [str(id)+'_vwap', str(id)+'_vol']
                # add ‘return’ column
                goods_df[str(id)+'_return'] = goods_df[str(id)+'_vwap'].rolling(2).apply(lambda x: x[1]/x[0])

                # Note that we fill the nan with 1.0 for the return value
                goods_df.fillna(1.0).to_csv(str(path + '/'+ str(id)+'_goods.csv'))
        else:
            print('No vwap_for_days_.csv !')
            return None

def data_preprocessing():
    
    if not os.path.exists('./newdata'):
        cleanData()

    if os.path.exists('./newdata/timeIndex_prcVwap_.csv'):
        prcVwap = pd.read_csv('./newdata/timeIndex_prcVwap_.csv', index_col=0, parse_dates = True)
        volume = pd.read_csv('./newdata/timeIndex_volume_.csv', index_col=0, parse_dates = True)
    else:
        print('No csv data !')

    if not os.path.exists('./newdata/vwap_for_days_.csv'):
        # make copy and fill in zeros
        vwap_data = prcVwap.copy()#.fillna(0.0)
        vol_data = volume.copy()#.fillna(0.0)

        vwap_all = get_vwap_volume_product(vwap_df=vwap_data, volume_df=vol_data)
        # determine days range
        time_index = vwap_data.index
        start_day = str(pd.Period(time_index[0], freq = 'D'))
        end_day = str(pd.Period(time_index[-1], freq = 'D'))


        day_list = pd.date_range(start_day, end_day, freq='D')

        vwap_for_days = pd.DataFrame([get_vwap_for_days(day, None, vwap_all, vol_data) for day in day_list], index = day_list)
        volume_days = pd.DataFrame([get_days_sum_data(vol_data, day) for day in day_list], index = day_list)

        print(vwap_for_days.shape)
        #print(vwap_for_days.head())
        vwap_for_days.to_csv('./newdata/vwap_for_days_.csv')
        volume_days.to_csv('./newdata/volume_for_days_.csv')

    # split_df_into_goods with return values
    split_df_into_goods_with_return()
#=================================

if __name__ == '__main__':
    data_preprocessing()

    vwap_for_days = pd.read_csv('./newdata/vwap_for_days_.csv', index_col=0, parse_dates = True)
    volume_days = pd.read_csv('./newdata/volume_for_days_.csv', index_col=0, parse_dates = True)

    print(vwap_for_days.shape)
    print(volume_days.shape)

    # plot the vwap for every products
    groups = list(range(29,30))
    print(groups)
    plt.figure()
    i = 1
    for group in groups:
        #plt.subplot(len(groups), 1, i)
        vwap_for_days[str(group)].dropna().plot(label=str(group), sharex=True)
        i += 1
    plt.legend()
    plt.show()

    id = 0
    goods = pd.read_csv('./newdata/goods/'+str(id)+'_goods.csv', index_col=0, parse_dates = True)
    goods[str(id)+'_return'].plot(style='-o')
    plt.show()
