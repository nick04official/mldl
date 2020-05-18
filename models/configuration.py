from configparser import ConfigParser 

"""
Class which allows reading tuning parameters from a .ini file
"""
class ParameterParser():

    """
    Default constructor
        Input:  - path .ini file
                - sections to check to be present in the .ini file
        Output: //
    """
    def __init__(self, path, assert_sections=["DATA_LOADER", "TRAINING", "OPTIMIZER"]):
    
        self.__config = ConfigParser()
        self.__config.read(path)
        self.__assert_sections = assert_sections

        for section in self.__assert_sections:
            try:
                self.__config[section]
            except KeyError:
                print("Section {} is not present in .ini file".format(section))

    """
    General function for retrieving parameters
        Input:  - section to get parameters from
        Output: - dictionary with parameters related to the section 
    """
    def __getparams__(self, section):
        ret = {}
        
        try:
            self.__config[section]
        except KeyError:
            return ret

        for param in list(self.__config[section]):
            ret[param] = self.__config[section][param]
        
        return ret

    """
    Function for retrieving the parameters of the training section
        Input:  //
        Output: - dictionary with parameters related to the section
    """
    def training(self):
        return self.__getparams__("TRAINING")

        """
    Function for retrieving the parameters of the optimizer section
        Input:  //
        Output: - dictionary with parameters related to the section
    """
    def optimizer(self):
        return self.__getparams__("OPTIMIZER")

        """
    Function for retrieving the parameters of the data loader section
        Input:  //
        Output: - dictionary with parameters related to the section
    """
    def data_loader(self):
        return self.__getparams__("DATA_LOADER")