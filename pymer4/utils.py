__all__  = ['get_resource_path']

__author__ = ['Eshin Jolly']
__license__ = "MIT"

from os.path import dirname,join, sep

def get_resource_path():
    """ Get path sample data directory. """
    return join(dirname(__file__), 'resources') + sep
