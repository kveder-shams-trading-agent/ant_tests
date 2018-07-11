from gym_seasonals.envs.seasonals_env import SeasonalsEnv
from gym_seasonals.envs.seasonals_env import portfolio_var
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas


class Agent:

    def __init__(self, env):
        self.env = env
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = env.reset()
        self.asset_ixs = range(len(asset_attributes['Return']))
        self.asset_histories = [[] for a in self.asset_ixs]
        self.last_tradable_events = tradable_events
        self.null_action = np.zeros(len(asset_attributes['Return']))

    def __update_histories(self, assets_attributes):
        self.asset_histories = [asset_history + [timestep] 
                for asset_history, timestep 
                in zip(self.asset_histories, assets_attributes)]

    def act(self):
        action = self.null_action
        observation, reward, done, info = self.env.step(action)
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        next_tradable_events = lookahead_events_calendar[1][1]
        self.__update_histories(
            [
                dict({'Level':asset_attributes['Level'][a]
                    },
                        **{'event_{}'.format(e):\
                                asset_attributes['Level'][a] 
                            if tradable_events[e] 
                            and self.last_tradable_events[e] 
                            and next_tradable_events[e] 
                            else None
                            for e in range(len(tradable_events))})
                    for a in self.asset_ixs])
        self.last_tradable_events = tradable_events
        return done

    def close(self):
        self.env.close()
        return [pandas.DataFrame(asset_history) 
                for asset_history in self.asset_histories]

def cusum_filter(df, h, asset_attribute='Level'):
    df['ema'] = df[asset_attribute].ewm(span=10).mean()
    S_pos = S_neg = 0
    filter_points = []
    last_ema = df['ema'][0]
    for ix, row in df.iterrows():
        S_pos = max(S_pos + row[asset_attribute] - last_ema, 0)
        S_neg = min(S_neg + row[asset_attribute] - last_ema, 0)
        if max(abs(S_neg), abs(S_pos)) > h:
            pt = row[asset_attribute]
            S_pos = S_neg = 0
        else: 
            pt = None
        filter_points.append(pt)
        last_ema = row['ema']
    return pandas.Series(filter_points)


def plot_df(df, colours=None):
    x_col = df.index
    colours = colours or ['b' for colname in df.columns]
    for colname, colour in zip(df.columns, colours):
        if colname != x_col.name and colour:
            plt.plot(x_col, 
                    df[colname], 
                    colour)
    plt.legend()
    plt.show()


def visualize_cusum(agent):
    done = False
    while not done:
        done = agent.act()
    dfs = agent.close()[4:]
    for ix in range(len(dfs)):
        dfs[ix]['cusum_sample'] = cusum_filter(dfs[ix], 7.5)
        plot_df(dfs[ix], colours=[
            'b', 'ro', 'yo', 'co', 'go', 'y--', 'k^'])
           

env = gym.make('seasonals-v1')
visualize_cusum(Agent(env))
