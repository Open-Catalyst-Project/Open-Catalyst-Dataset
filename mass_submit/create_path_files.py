import pickle
import os
import sys

script_list = []
for subdir, dirs, files in os.walk('./'):
    for file in files:
        if file == 'SUB_vasp.sh':
            script_name = os.path.join(subdir)
            script_list.append(script_name)

print(len(script_list))
with open('samples_to_run.pkl', 'wb') as f:
    pickle.dump(script_list, f)
