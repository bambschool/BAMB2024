import numpy as np
import scipy.io as scio
import pandas as pd


subject = 0 

# 
mat = scio.loadmat('../Downloads/Haith_data/Expt1_Data.mat')


data = mat['data'][0][subject][0][0]

# sampling freq, 130 Hz, in meters
# start location is at x=.6, y=.23
x = np.array([d.squeeze() for d in data[0][0]])
y = np.array([d.squeeze() for d in data[1][0]])

x -= np.mean(x, axis=0)[0]
y -= np.mean(y, axis=0)[0]

xy = np.array([x,y])

df = pd.DataFrame()

df['jump_time'] = np.squeeze(data[4]) 	# jumpt time in s

df['target_id_pre'] = np.squeeze(data[2]) # initial and post-jump target location, 8 cm from start
df['target_id_post'] = np.squeeze(data[3]) 	# 0 means east, 1 north-east, ... until 7

# calculate target locations in cm

# get angles
all_ang = np.linspace(0, 2*np.pi, 8, endpoint=False)

df['target_angle_pre'] = [all_ang[t] for t in df.target_id_pre.values]
df['target_angle_post'] = [all_ang[t] for t in df.target_id_post.values]

# get x = radius * cos(angle), y = radius * sin(angle)
df['target_x_pre'] = .08*np.cos(df['target_angle_pre'])
df['target_y_pre'] = .08*np.sin(df['target_angle_pre'])

df['target_x_post'] = .08*np.cos(df['target_angle_pre'])
df['target_y_post'] = .08*np.sin(df['target_angle_pre'])

# save stuff
np.save('Haith_subj-%.0f_xy.npy' %subject, xy)
df.to_pickle('Haith_subj-%.0f_targets.p' %subject)