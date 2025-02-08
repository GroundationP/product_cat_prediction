import pytest
import sys
import os#
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/app"))
sys.path.append(src_path) #  Add 'src' to Python path
import utils_and_constants # import data_load, wordcloud
import wordcloud



#  Load the CSV file
PATH = "/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/data/original/"
files = ['X_train_update.csv', 'Y_train_CVw08PX.csv', 'X_test_update.csv']

@pytest.fixture
def read_file():
    '''Fonction to load data
    '''
    return utils_and_constants.data_load(PATH, files)
    
    
def test_read_file(read_file):
    '''Checking if train/test and corpus files are not empty and respect dict/string formats
    '''
    # Check if train file is a non empty dict
    assert isinstance(read_file[0][0], dict)
    assert len(read_file[0][0]['data']) > 0
    # Check if test file is a non empty dict
    assert isinstance(read_file[1][0], dict)
    assert len(read_file[1][0]['data']) > 0
    # Check if trainset > testset
    assert len(read_file[0][0]['data']) > len(read_file[1][0]['data'])
    # Check if corpus_desig_json is a non empty string
    assert isinstance(read_file[2][0], str)
    assert len(read_file[2]) > 0
