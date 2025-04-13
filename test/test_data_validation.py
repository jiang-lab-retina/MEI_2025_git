import unittest
import sys
from os import path
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import jianglab as jl
import pickle
import os
from scipy.signal import find_peaks

class Test_correlation_study(unittest.TestCase):
    def setUp(self):        
        file_path = os.path.join("raw_data", "2025.03.19-16.26.24-Rec.pkl")
        with open(file_path, "rb") as file:
            pkl_data = pickle.load(file)
        self.pkl_data = pkl_data
        
        
    def test_data_structure_meta_data(self):
        jl.dict_to_tree(self.pkl_data["meta_data"]).show()
        
    def test_data_structure_units_data(self):
        jl.dict_to_tree(self.pkl_data["units_data"]["1"]).show()
        
    def test_light_ref(self):
        light_ref = self.pkl_data["meta_data"]["light_reference_raw"]
        plt.plot(light_ref[0])
        plt.plot(light_ref[1])
        plt.show()
        
    def test_npy_data(self):
        npy_data = np.load(os.path.join("ILSVRC2012_npy", "ILSVRC2012_test_00000300_00000001_00000300.npy"))
        print(npy_data.shape)
        plt.plot(npy_data.mean(axis=(1,2)))
        plt.title(npy_data.shape)
        plt.show()
        
    def test_validate_frame_number(self):
        light_ref_raw = self.pkl_data["meta_data"]["light_reference_raw"]
        ligh_ref_peak, _ = find_peaks(np.diff(light_ref_raw[1]),
                                   distance=10,
                                   height=10000
                                   )
        plt.plot(light_ref_raw[1])
        plt.plot(ligh_ref_peak+2, light_ref_raw[1][ligh_ref_peak+2], "ro")
        # display the peak index number on top the dot
        for i, peak in enumerate(ligh_ref_peak):
            if i % 10 == 0:
                plt.text(peak+2, light_ref_raw[1][peak+2], str(i), color="black")
        plt.show()
        # print(ligh_ref_peak)
        # npy_data = np.load(os.path.join("ILSVRC2012_npy", "ILSVRC2012_test_00000300_00000001_00000300.npy"))
        # npy_data_mean = npy_data.mean(axis=(1,2))
        # plt.plot(npy_data_mean)
        # plt.show()
        
    
        

        
if __name__ == "__main__":
    unittest.main()
    
    