import os
import configparser

config = configparser.ConfigParser()
config.read("conf/xfmkit.conf")

def get(section, value, default=None, mandatory=False):
    """
    Reads config value from within section
    or return default
    """

    try:
        return config.get(section, value).strip()
    
    except:
        if mandatory: 
            raise Exception(f"Mandatory value {value} not found in section {section}")
        else:
            return default