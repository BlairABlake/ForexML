import os
import pandas as pd
from forexenv import ForexTradingEnv
from models.dqn import DQNAgent
import gc

historical_data = pd.read_csv('data2010_usdjpy.csv', names=['date', 'Open', 'High', 'Low', 'Close'])

if len(str(historical_data['date'][1])) == 14:
    historical_data['date'] = historical_data['date'].astype(str)
    datetime_series = pd.to_datetime(historical_data['date'].str.slice(0, 8) + ' ' + historical_data['date'].str.slice(8, 10) + ':' + historical_data['date'].str.slice(10, 12), format='%Y%m%d %H:%M')
    historical_data['date'] = datetime_series.dt.strftime('%Y/%m/%d %H:%M')
    #data['date'] = data['date'].astype(str).apply(lambda x: pd.to_datetime(x[:8] + ' ' + x[8:10] + ':' + x[10:12], format='%Y%m%d %H:%M').strftime('%Y/%m/%d %H:%M'))
else:
    historical_data['date'] = pd.to_datetime(historical_data['date'])

historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data.set_index('date', inplace=True)

historical_data = historical_data.resample('2H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

duration = 40
episode_start = 46
episodes = 100
batch_size = 16

env = ForexTradingEnv(historical_data, duration=duration, cash=1000000)

# Initialize the DQN agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size, duration=duration)
model_path = "weights_{:04d}.hdf5".format(episode_start)
if(os.path.exists(model_path)):
    agent.load(model_path)



for e in range(episode_start + 1, episodes):
    done = False
    time=0
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        gc.collect()
        if done:
            agent.save("weights_" + "{:04d}".format(e) + ".hdf5")
            report = env.backtest.make_report()
            env.backtest.plot(open_browser=False)
            with open(f"trading_report_{e}.txt", mode="w") as f:
                f.write(str(report))
            print(f"Episode: {e+1}/{episodes}, Score: {time+1}")
        elif len(agent.memory) > batch_size:
            agent.replay(batch_size)