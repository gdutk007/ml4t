import datetime as dt
import os
import io
from io import TextIOWrapper
import re
import numpy as np
import math
import pandas as pd
from util import get_data, plot_data
from TheoreticallyOptimalStrategy import *



if __name__ == "__main__":
    start_date=dt.datetime(2008,1,1)
    end_date=dt.datetime(2009,12,31)
    tosTrades = testPolicy("JPM",start_date,end_date, 1000000)
    portVals = compute_portvals(tosTrades, start_val=sv,commission=9.95, impact=impact)