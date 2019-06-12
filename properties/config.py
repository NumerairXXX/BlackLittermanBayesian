import configparser

def get_config(env):
    config = configparser.RawConfigParser()
    config.add_section(env)

    data_dir = "fake direction here"
    config.set(env,'data_folder',data_dir)
    return configparser