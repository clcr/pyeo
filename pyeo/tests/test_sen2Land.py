import Sen2Land
import configparser
import os

def test_sen2cor_process_windows():
    #Tests that Sen2Cor throws an error message
    conf = configparser.ConfigParser()
    conf.read(os.path.join("..", "conf.ini"))
    target = "made/up/filepath"
    out = Sen2Land.sen2cor_process_windows(target, conf)
    assert False