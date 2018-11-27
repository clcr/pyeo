from apps.proc_steps import Sen2Land
import configparser
import os

def test_sen2cor_process_windows():
    #Tests that Sen2Cor throws an error message
    conf = configparser.ConfigParser()
    conf.read(os.path.join("..", "conf.ini"))
    target = r"C:\Maps\aois\kenya\S2A_MSIL1C_20171223T075319_N0206_R135_T36NXG_20171223T113022.SAFE"
    out = Sen2Land.sen2cor_process_windows(target, conf)
    assert out == -1


def test_sen2cor_process():
    pass


