from datetime import datetime
from pytz import timezone
import os
import shutil
import torch
import time

class pttm():
    def __init__(self):
        self.start_time = time.time()
        self.previos_time = self.start_time

    def print_status(self, epoch, idx, trainloader):
        _len = len(trainloader)
        current_time = time.time()
        time_step = current_time - self.previos_time
        self.previos_time = current_time

        remain_time = time.strftime('%H:%M:%S', time.gmtime(int((len(trainloader) - (idx + 1)) * time_step)))
        progress_time = time.strftime('%H:%M:%S', time.gmtime(int(current_time - self.start_time)))

        print("Epoch : {} [{}/{} ({}%)] [{}/{}]   {}{}".format(
                                                epoch, 
                                                idx + 1, 
                                                _len, 
                                                int((idx + 1)/len(trainloader) * 100),
                                                progress_time,
                                                remain_time,
                                                "\033[101m" + " " * int((idx + 1)/len(trainloader) * 30) + "\033[0m", "\033[43m" + " " * (30 - int((idx + 1)/len(trainloader) * 30)) + "\033[0m"), 
                                                end="\r"
                                            )

def get_current_time():
    return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')

def make_new_work_space():
    current_time = get_current_time()
    new_root_dir = os.path.join("./log", current_time)
    
    if not os.path.exists(new_root_dir):
        os.makedirs(new_root_dir)

    return new_root_dir

def save_model(root_dir, epoch, model, name):
    model_dir = os.path.join(root_dir, "model")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict()
            }, model_dir + "/{}_model.pth".format(name))

def save_config_file(root_dir):
    config_file = "./config.py"
    log_dir = os.path.join(root_dir, "train")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.isfile(log_dir + "/config.py"):
        pass
    else:
        shutil.copy(config_file, log_dir)

def save_testing_log(root_dir, msg):
    log_dir = os.path.join(root_dir, "test")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    f = open(log_dir + "/test_log.txt", 'a')
    f.write(msg + "\n")
    f.close()


def copy_result(work_dir, valid_name):
    for (path, _, files) in os.walk(os.path.join(work_dir, "buffer", "total", valid_name)):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                if not os.path.exists(path.replace("buffer", "result")):
                    os.makedirs(path.replace("buffer", "result"))
                shutil.copy("%s/%s" % (path, filename), "%s/%s" % (path.replace("buffer", "result"), filename))
    
    for (path, _, files) in os.walk(os.path.join(work_dir, "buffer", "pred", valid_name)):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                if not os.path.exists(path.replace("buffer", "result")):
                    os.makedirs(path.replace("buffer", "result"))
                shutil.copy("%s/%s" % (path, filename), "%s/%s" % (path.replace("buffer", "result"), filename))
    
    for (path, _, files) in os.walk(os.path.join(work_dir, "buffer", "gt", valid_name)):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                if not os.path.exists(path.replace("buffer", "result")):
                    os.makedirs(path.replace("buffer", "result"))
                shutil.copy("%s/%s" % (path, filename), "%s/%s" % (path.replace("buffer", "result"), filename))
