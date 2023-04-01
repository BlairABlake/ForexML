from backtesting import *
from backtesting._stats import compute_stats
from backtesting._util import _Indicator, _Data, try_

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

class _OutOfMoneyError(Exception):
    pass

class CustomStrategy(Strategy):
    def init(self):
        self._count = 0
        self._action_mapping = [self.buy, self.sell, self._stay]
        
    def next(self, action):
        self._action_mapping[action]()
        self._count += 1
    
    def _stay(self): pass
    
    
class CustomBacktest(Backtest):
    m_step = 0
    
    def __init__(self,
                 data: pd.DataFrame,
                 duration=10,
                 **kwargs):
        strategy = CustomStrategy
        super().__init__(data,strategy,**kwargs)
        
        self.m_duration = duration
        self.m_step += self.m_duration
        

    
    def setup(self, **kwargs):
        self.m_data = _Data(self._data.copy(deep=False))
        self.m_broker = self._broker(data=self.m_data)
        self.m_strategy: Strategy = self._strategy(self.m_broker, self.m_data, kwargs)

        self.m_strategy.init()
        self.m_data._update()  # Strategy.init might have changed/added to data.df

        # Indicators used in Strategy.next()
        self.m_indicator_attrs = {attr: indicator
                           for attr, indicator in self.m_strategy.__dict__.items()
                           if isinstance(indicator, _Indicator)}.items()

        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        self.m_start = 1 + max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                         for _, indicator in self.m_indicator_attrs), default=0)
        
        # self.m_step should always be in the range [self.duration, len(self._data) - self._duration]
        self.m_step = max(self.m_duration, self.m_start)
        
    def next(self, action, **kwargs):
        with np.errstate(invalid='ignore'):
            done = False
            self.m_data._set_length(self.m_step + 1)
            for attr, indicator in self.m_indicator_attrs:
                # Slice indicator on the last dimension (case of 2d indicator)
                setattr(self.m_strategy, attr, indicator[..., :self.m_step + 1])

            # Handle orders processing and broker stuff
            try:
                self.m_broker.next()
            except _OutOfMoneyError:
                # Close any remaining open trades so they produce some stats
                for trade in self.m_broker.trades:
                    trade.close()

                # Re-run broker one last time to handle orders placed in the last strategy
                # iteration. Use the same OHLC values as in the last broker iteration.
                if self.m_start < len(self._data):
                    try_(self.m_broker.next, exception=_OutOfMoneyError)
                    done = True

            # Next tick, a moment before bar close
            self.m_strategy.next(action)
        
        self.m_step += 1
        # self.m_step should always be in the range [self.duration, len(self._data) - self._duration]
        done = self.m_step >= len(self._data) - self.m_duration if not done else done
        
        tradings = list(filter(lambda trade: trade.exit_price is None, self.m_broker.trades))
        reward = self.m_broker.equity
        return self._current_ohlc(), done, reward

    def _current_ohlc(self):
        return np.reshape(
            np.array([self.m_data.Open[self.m_step-self.m_duration:self.m_step], self.m_data.High[self.m_step-self.m_duration:self.m_step], self.m_data.Low[self.m_step-self.m_duration:self.m_step], self.m_data.Close[self.m_step-self.m_duration:self.m_step]]),
            (self.m_duration, 4)
        )
    
    def make_report(self):
        self.m_data._set_length(len(self._data))

        equity = pd.Series(self.m_broker._equity).bfill().fillna(self.m_broker._cash).values
        self._results = compute_stats(
            trades=self.m_broker.closed_trades,
            equity=equity,
            ohlc_data=self._data,
            risk_free_rate=0.0,
            strategy_instance=self.m_strategy,
        )

        return self._results
        
        
class ForexTradingEnv(gym.Env):
    def __init__(self, data, duration=10, cash=1000000, exclusive_orders=True):
      super().__init__()
      self._data = data
      self.duration = duration
      self.cash = cash
      self.exclusive_orders = exclusive_orders
      
      # 3 actions, corresponding to Buy, Sell, Stay respectively
      self.action_space = spaces.Discrete(3)
      self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    
    def reset(self):
      self.backtest = CustomBacktest(self._data, duration=self.duration, cash=self.cash, commission=.002, exclusive_orders=self.exclusive_orders)
      self.backtest.setup()
      return self.backtest._current_ohlc()
    
    def step(self, action):
        state, done, reward = self.backtest.next(action)
        return state, reward, done, {}