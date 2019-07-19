
'''
Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li
A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances
IMWUT/UbiComp 2019
---------------------------------------------------------------------------------------
'''

from datetime import datetime

class Utility():

    def __init__(self):
        pass

    @staticmethod
    def get_elapsed_seconds(start_time, end_time):
        return (end_time - start_time).total_seconds()

    @staticmethod
    def string2time(string_time):

        _year, _month, _day, _hour, _minute, _second = Utility.split_time_1(
            string_time)

        return datetime(_year, _month, _day, _hour, _minute, _second)

    @staticmethod
    def split_time_1(string_time):  # string_time format: 2016-12-31 15:18:24
        t = string_time.split(' ')
        if len(t) == 1:
            print(string_time)

        t1 = t[0].split('-')
        t2 = t[1].split(':')
        year = int(float(t1[0]))
        month = int(float(t1[1]))
        day = int(float(t1[2]))
        hour = int(float(t2[0]))
        minute = int(float(t2[1]))
        second = int(float(t2[2]))

        return year, month, day, hour, minute, second

