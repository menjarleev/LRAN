import os

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

    @staticmethod
    def log_print(opt, log):
        print(log)
        if not opt.debug:
            log_path = os.path.join(opt.ckpt_root, opt.name, 'log.txt')
            with open(log_path, "a") as log_file:
                log_file.write('%s\n' % log)

