import glob
import numpy as np
import pandas as pd
import numba

def make_array(data_list):
    """
    Make arrays out of string data
    Args:
        data_list(list[strings])
            Given a list of the parsed strings
    Returns:
        data_list(list(float, bool))
            Returns a list of tuples as (time till catastrophy, labeling)
    """
    for i in range(len(data_list)):
        data = data_list[i]
        time_start = data.index(",")+1
        time_end = find_nth(data, ",", 2)
        fluor_start = time_end+1
        
        time = float(data[time_start:time_end])
        fluor = True
        if data[fluor_start:] == "False":
            fluor = False
            
        data_list[i] = (time, fluor)
    return data_list

def find_nth(string, substring, n):
    """Finds the index of the nth instance of the given substring from the given string"""
    location = string.index(substring)
    while n > 1:
        location = string.index(substring, location+len(substring))
        n -= 1
    return location

def ecdf_organizer(data):
    """
    Organizes the conversion of the list of tuples into the ecdf values
    Args:
        data(list(float, bool))
            list of tuples as (time till catastrophy, labeling)
    Returns: 
        ecdf_lists(list(np.array(float), np.array(float)))
            list of two np.arrays of floats for the percentage of values below a value
    """
    labeled, unlabeled = split_by_label(data)
    
    labeled = np.sort(labeled)
    unlabeled = np.sort(unlabeled)
    labeled_len, unlabeled_len, high = get_important_vals(labeled, unlabeled)
   
    labeled_ecdf = ecdf_values(labeled, labeled_len, high)
    unlabeled_ecdf = ecdf_values(unlabeled, unlabeled_len, high)
    
    ecdf_lists = [labeled_ecdf, unlabeled_ecdf]
    return ecdf_lists
    
def split_by_label(array):
    """
    Split the arrays into two depending on the label boolean
    Args:
        data(list(float, bool))
            list of tuples as (time till catastrophy, labeling)
    Returns:
        labeled_np, unlabeled_np (np.arrays of floats)
    """
    labeled_list = []
    unlabeled_list = []
    for data in array:
        if data[1]:
            labeled_list.append(data[0])
        else:
            unlabeled_list.append(data[0])
    labeled_np = np.array(labeled_list)
    unlabeled_np = np.array(unlabeled_list)
    return labeled_np, unlabeled_np

def get_important_vals(labeled, unlabeled):
    """
    Get values that are important for later
    Args: 
        labeled_np (np.array of floats)
            list of numbers for catastrophy times
        unlabeled_np (np.array of floats)
            list of numbers for catastrophy times
            
    Returns:
        labeled_len (int)
            max indexable number for labeled list
        unlabeled_len
            max indexable number for unlabeled list
        high
            highest value in either list
    """
    labeled_len = len(labeled) - 1
    unlabeled_len = len(unlabeled) - 1
    high = 0
    if labeled[labeled_len] > unlabeled[unlabeled_len]:
        high = labeled[labeled_len]+1
    else:
        high = unlabeled[unlabeled_len]+1
    return [labeled_len, unlabeled_len, int(high)]

def ecdf_values(data, length, high):
    """
    Converts data from original data to ecdf values
    Args:
        data (np.array of floats)
            floats of all times of catastrophy in numerical order
        length (int)
            highest indexable element in data
        high (int)
            highest number in general to iterate up to (may or may not be close to length)
    """
    ecdf_list = np.zeros(high)
    pos = 0
    for i in range(high):
        cat_time = data[pos]
        while pos < length and cat_time <= i:
            pos += 1
            cat_time = data[pos]
        ecdf_list[i] = pos/(length+1)
        if i >= cat_time:
            ecdf_list[i] = 1
        
    return ecdf_list