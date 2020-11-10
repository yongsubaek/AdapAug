import copy, os
import logging
import warnings
from ray import tune
from theconf import Config as C

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "DeprecationWarning: 'saved_variables' is deprecated", UserWarning)

class CustomLogger(tune.logger.Logger):
    def __init__(self):
        base_path = f"models/{C.get()['exp_name']}"
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), base_path)
        self._file = open(base_path + "/log.csv", "w")
        self._first = True
    def on_result(self, result):
        if self._first:
            self._file.write(",".join(result.keys()) + "\n")
            self._first = False
        txt = ",".join([v for k,v in result.items()]) + "\n"
        self._file.write(txt)

    def close(self):
        self._file.close()

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath, level=logging.DEBUG):
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def __len__(self):
        return len(self.shadow)

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            # see : https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/ExponentialMovingAverage?hl=PL
            mu = min(self.mu, (1. + step) / (10 + step))

        for name, x in module.state_dict().items():
            if name in self.shadow:
                new_average = (1.0 - mu) * x + mu * self.shadow[name]
                self.shadow[name] = new_average.clone()
            else:
                self.shadow[name] = x.clone()
