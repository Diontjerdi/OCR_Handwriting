import configparser

def process_config(conf_file):
    """process configure file to generate CommonParams, DataSetParams, NetParams, TrainingParams
    Args:
        conf_file: configure file path
    Returns:
        CommonParams, DataSetParams, NetParams, TrainingParams
    """
    common_params = {}
    dataset_params = {}
    net_params = {}
    training_params = {}

    # configure_parser
    config = configparser.ConfigParser()
    config.read(conf_file)

    # sections and options
    for section in config.sections():
        # construct common_params
        if section == 'Common':
            for option in config.options(section):
                common_params[option] = config.get(section, option)
        # construct dataset_params
        if section == 'DataSet':
            for option in config.options(section):
                dataset_params[option] = config.get(section, option)
        # construct net_params
        if section == 'Net':
            for option in config.options(section):
                net_params[option] = config.get(section, option)
        # construct training_params
        if section == 'Training':
            for option in config.options(section):
                training_params[option] = config.get(section, option)

    return common_params, dataset_params, net_params, training_params


if __name__ == "__main__":
    conf_file = r"conf/train.cfg"
    common_params, dataset_params, net_params, training_params = process_config(conf_file)
    print(common_params, dataset_params, net_params, training_params)