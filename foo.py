from base import dispatcher
from FSM import FSM
from tasks import lp_task_list

import os
import re
import time

while(True):
    usage = int(re.sub('\D', '', os.popen('nvidia-smi | sed -n 10p').read().split('|')[3]))
    if(usage > 20):
        print('Sleeping......')
        time.sleep(60)
    else:
        break

top = FSM()
globalDispatcher = dispatcher(top, taskList=lp_task_list)
globalDispatcher.run()