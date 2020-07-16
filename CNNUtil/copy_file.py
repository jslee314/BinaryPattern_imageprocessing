import os
import shutil

src = 'C:/Users/oprime/Downloads/json_20200629_right0147242191_5f900470-0431-4fce-8120-194d1c934fc5.json'
data_path = 'C:/Users/oprime/Downloads/조원태'
count = 0

for (root, dirs, files) in os.walk(data_path):
    for dir in dirs:
        shutil.copy(src, os.path.join(root, dir) + '/')


