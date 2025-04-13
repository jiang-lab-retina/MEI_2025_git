import sys
from os import path
import os
#sys.path.insert(1, '../')

#load the path to the directory containing the package

#sys.path.append(path.dirname(path.dirname(os.getcwd())))
import jianglab as jl
import McsPy.McsCMOSMEA as McsCMOSMEA
#from load_raw_data import load_basic_cmtr_data
#from feature_analysis import *
import matplotlib.pyplot as plt
from MEI_analysis_helper import *
import glob
import pickle
import pandas as pd

def plot_sta(pkl_data, 
             movie_data_array,
             section_time_frame_num = [184, np.inf]):  
     
    acq_rate = pkl_data["meta_data"]["acquisition_rate"]
    frame_time = pkl_data["meta_data"]["frame_time"]
    
    for unit_data in pkl_data["units_data"].values():
        print(unit_data["unitID"], unit_data["row"], unit_data["col"])

        all_raw_spike_ns = unit_data["raw_spike"]
        # the values are in nanosecond
        all_raw_spike = (np.array(all_raw_spike_ns) * acq_rate / 1_000_000).astype(np.int64) 
        #convert to 20khz unit
        all_raw_spike_frame_num =convert_timestamp_to_frame_num(all_raw_spike, frame_time)
        raw_spike = all_raw_spike_frame_num[(all_raw_spike_frame_num > section_time_frame_num[0]) \
                                        & (all_raw_spike_frame_num < section_time_frame_num[1])]
        raw_spike = raw_spike - section_time_frame_num[0]

        sta = get_sta(trace_to_analyze = raw_spike, 
                movie_data_array = movie_data_array, 
                cover_range = (-600, 200), 
                trace_type = "timestamp")
        
        jl.plot_3d_array(sta, plot_min = sta.min(), plot_max = sta.max(), interval = 10,
                            title = "Unit " + str(unit_data["unitID"]))
        plt.show()
        
        data_1d = sta.mean(axis = (1,2))
        extreme_coordinate = np.unravel_index(np.argmax(np.abs(sta-sta.mean())), sta.shape)
        plt.plot(sta[:, extreme_coordinate[1], extreme_coordinate[2]], c = "r")
        plt.show()
        
def compress_image_data(image_data_array_name,
                        image_interval = 60,
                        image_duration = 30,
                        image_start_time = 0,
                        show_plot = True,
                        save_compressed_image_data = True,
                        compressed_image_folder_name = "compressed_ILSVRC2012",
                        ):
    image_data_array = np.load(image_data_array_name)
    output_file_name = path.basename(image_data_array_name).split(".")[0] + "_compressed.npy"
    for i in range(0, image_data_array.shape[0], image_interval):
        image_data = image_data_array[i:i+image_duration, :, :]
        image_data_mean = image_data.mean(axis = 0)
        if np.any(image_data[0] != image_data_mean) or np.any(image_data[image_duration-1] != image_data_mean):
            print("image data is not consistent")
            return
    if "00000001" in path.basename(image_data_array_name):
        compressed_image_data = image_data_array[30:][::60]
    else:
        compressed_image_data = image_data_array[::60]
    if save_compressed_image_data:
        if not path.exists(compressed_image_folder_name):
            os.makedirs(compressed_image_folder_name, exist_ok = True)
        np.save(path.join(compressed_image_folder_name, output_file_name), compressed_image_data)
    if show_plot:
        #plot first 100 images in subplots
        plt.plot(image_data_array.mean(axis = (1,2)))
        plt.suptitle(path.basename(image_data_array_name).split(".")[0])
        plt.show()
        fig, axs = plt.subplots(10, 10)
        for i in range(100):
            axs[i//10, i%10].imshow(image_data_array[i*60])
        plt.suptitle(path.basename(image_data_array_name).split(".")[0])
        plt.show()
        
        fig, axs = plt.subplots(10, 10)
        for j in range(100):
            axs[j//10, j%10].imshow(image_data_array[j*60+30-1])
        plt.suptitle(path.basename(image_data_array_name).split(".")[0] + " + 30 frames")
        plt.show()  
    
    return compressed_image_data

def get_absolute_time_of_image_data(pkl_file_name,
                                                 image_data_array_name,
                                                 pre_start_frame_num = 184,
                                                ):
    with open(pkl_file_name, "rb") as f:
        pkl_data = pickle.load(f)
    frame_time = pkl_data["meta_data"]["frame_time"]
    onset_time = frame_time[pre_start_frame_num]
    image_data_array = np.load(image_data_array_name)
    total_frame_num = image_data_array.shape[0]
    end_frame_num = pre_start_frame_num + total_frame_num
    offset_time = frame_time[end_frame_num]
    return onset_time, offset_time


def get_connected_image_to_firing_time(firing_time_stamps,
                                       frame_time,
                                        compressed_image_data,
                                        pre_start_frame_num = 184,
                                        acq_rate = 20_000,
                                        ):
    output_dict = {}

    firing_time_stamps = np.array(firing_time_stamps) #unit in nanosecond
    # convert to 20khz unit
    firing_time_stamps = firing_time_stamps / 1_000_000 * acq_rate  #unit in 20khz
    for i, image in enumerate(compressed_image_data):
        image_frame_start_index = pre_start_frame_num + i*60
        image_frame_end_index = image_frame_start_index + 60
        image_frame_start_time = frame_time[image_frame_start_index]
        image_frame_end_time = frame_time[image_frame_end_index]
        firing_time_stamps_in_image_frame = firing_time_stamps[(firing_time_stamps > image_frame_start_time) \
                                                                & (firing_time_stamps < image_frame_end_time)]
        firing_time_stamps_in_image_frame = firing_time_stamps_in_image_frame - image_frame_start_time
        duration = image_frame_end_time - image_frame_start_time
        output_dict[i] = {"firing_time_stamps": firing_time_stamps_in_image_frame,
                            "start_time": image_frame_start_time,
                            "end_time": image_frame_end_time,
                            "duration": duration,
                            }
    return output_dict
        
def add_connected_image_to_firing_time_to_pkl(pkl_file_name,
                                                compressed_image_data_name_list,
                                                image_data_name_list,
                                                pre_start_frame_num = 181, #184
                                                inter_set_frame_num_list = [361, 361, 363],
                                                acq_rate = 20_000,
                                                ):
    with open(pkl_file_name, "rb") as f:
        pkl_data = pickle.load(f)
    frame_time = pkl_data["meta_data"]["frame_time"]
    image_data_frame_num = 0
    for i, (compressed_image_data_name, image_data_name) in enumerate(zip(compressed_image_data_name_list, image_data_name_list)):
        compressed_image_data = np.load(compressed_image_data_name)
        if i == 0:
            set_start_frame_num = pre_start_frame_num
        else:
            set_start_frame_num = set_start_frame_num + image_data_frame_num + inter_set_frame_num_list[i-1]
        image_data = np.load(image_data_name)
        image_data_frame_num = image_data.shape[0]

        for unit_data in pkl_data["units_data"].values():
            firing_time_stamps = unit_data["raw_spike"]
            image_index_str = path.basename(compressed_image_data_name).split("_")[3:5]
            image_index_str = "_".join(image_index_str)
            
            connected_image_to_firing_time = get_connected_image_to_firing_time(firing_time_stamps,
                                                                                frame_time,
                                                                                compressed_image_data,
                                                                                set_start_frame_num,
                                                                                acq_rate)
            if "connected_image_to_firing_time" not in unit_data:
                unit_data["connected_image_to_firing_time"] = {}
            unit_data["connected_image_to_firing_time"][image_index_str] = connected_image_to_firing_time
    output_pkl_file_name = pkl_file_name.split(".")[0] + "_connected_image_to_firing_time.pkl"
    with open(output_pkl_file_name, "wb") as f:
        pickle.dump(pkl_data, f)

def validate_connected_image_to_firing_time(connected_pkl_file_name):
    with open(connected_pkl_file_name, "rb") as f:
        pkl_data = pickle.load(f)
    total_unit_num = len(pkl_data["units_data"])
    row_num = 12
    col_num = total_unit_num // row_num + 1
    
    fig, axs = plt.subplots(row_num, col_num)
    for i, unit_data in enumerate(pkl_data["units_data"].values()):
        trace_list = []
        for image_index_range_str, connected_image_to_firing_time in unit_data["connected_image_to_firing_time"].items():
            for image_index, connected_image_to_firing_time_dict in connected_image_to_firing_time.items():
                firing_time_stamps = connected_image_to_firing_time_dict["firing_time_stamps"] 
                duration = connected_image_to_firing_time_dict["duration"]
                #convert to 10 hz firing rate with 100ms bin
                firing_rate_list = np.histogram(firing_time_stamps, bins = np.arange(0, 20000, 2000))[0]
                trace_list.append(firing_rate_list)
        trace_list = np.array(trace_list)
        #plt.plot(trace_list.T, alpha = 0.1)
        axs[i//col_num, i%col_num].plot(trace_list.mean(axis = 0))
    #plt.plot(trace_list.std(axis = 0))
    plt.suptitle(trace_list.shape)
    plt.show()
    
def validate_connection_by_light_ref_plot(connected_pkl_file_name):
    with open(connected_pkl_file_name, "rb") as f:
        pkl_data = pickle.load(f)
    total_unit_num = len(pkl_data["units_data"])
    row_num = 12
    col_num = total_unit_num // row_num + 1
    light_ref = pkl_data["meta_data"]["light_reference_raw"]
    light_ref = light_ref / 1_000_000 * 20_000
    plt.plot(light_ref[0], alpha = 0.5, linewidth = 0.5)
    plt.plot(light_ref[1], alpha = 0.5, linewidth = 0.5)
    start_time_list, end_time_list = [], []
    unit_data = list(pkl_data["units_data"].values())[0]
    for image_index_range_str, connected_image_to_firing_time in unit_data["connected_image_to_firing_time"].items():
        for image_index, connected_image_to_firing_time_dict in connected_image_to_firing_time.items():
            start_time_list.append(connected_image_to_firing_time_dict["start_time"])
            end_time_list.append(connected_image_to_firing_time_dict["end_time"])
    print(start_time_list)
    print(end_time_list)
    plt.scatter(start_time_list, 0* np.ones(len(start_time_list)), c = "r")
    plt.scatter(end_time_list, 50* np.ones(len(end_time_list)), c = "b")
    plt.show()
    
    
def make_simple_dataset(connected_pkl_file_name):
    with open(connected_pkl_file_name, "rb") as f:
        pkl_data = pickle.load(f)
    
    # Prepare data collection for better performance
    data_dict = {}
    unit_ids = []
    image_ids = set()
    
    # First pass: collect all data
    for key, unit_data in pkl_data["units_data"].items():
        unit_ids.append(key)
        unit_dict = {}
        
        for image_index_range_str, connected_image_to_firing_time in unit_data["connected_image_to_firing_time"].items():
            for image_index, connected_image_to_firing_time_dict in connected_image_to_firing_time.items():
                firing_time_stamps = connected_image_to_firing_time_dict["firing_time_stamps"]
                # Convert to 10 hz firing rate with 100ms bin
                firing_rate_list = np.histogram(firing_time_stamps, bins=np.arange(0, 20000, 2000))[0]
                image_id = f"{image_index_range_str}--{image_index}"
                image_ids.add(image_id)
                unit_dict[image_id] = firing_rate_list
        
        data_dict[key] = unit_dict
    
    # Create DataFrame with all data at once
    # Initialize an empty DataFrame with the right structure
    columns = sorted(list(image_ids))
    index = unit_ids
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill in the DataFrame
    for unit_id, unit_data in data_dict.items():
        for image_id, firing_rates in unit_data.items():
            df.loc[unit_id, image_id] = firing_rates.tolist()
    
    # Save to CSV
    output_file = os.path.basename(connected_pkl_file_name).split(".p")[0] + ".csv"
    df.to_csv(output_file)
    
    return df
    
def find_cell_center(simple_dataset_path):
    df = pd.read_csv(simple_dataset_path)
    compressed_image_data_names = glob.glob("compressed_ILSVRC2012/*.npy")
    compressed_image_data_names.sort()
    total_unit_num = len(df)
    row_num = 12
    col_num = total_unit_num // row_num + 1 
    all_images_data = []
    for compressed_image_data_name in compressed_image_data_names:
        compressed_image_data = np.load(compressed_image_data_name)
        all_images_data.extend(compressed_image_data)
    all_images_data = np.array(all_images_data)
    all_images_mean = all_images_data.mean(axis = 0)
    np.save("all_images_mean.npy", all_images_mean)
    
    
    final_X_dict = {}
    final_Y_dict = {}
    receptive_field_dict = {}
    fig, axs = plt.subplots(row_num, col_num)
    for k, unit_id in enumerate(df.index):
        X, Y = [], []
        sta_result = []
        firing_rates_list = []
        for i, compressed_image_data_name in enumerate(compressed_image_data_names):
            compressed_image_data = np.load(compressed_image_data_name)
            compressed_image_batch_name = compressed_image_data_name.split("_")[-3:-1]
            compressed_image_batch_name = "_".join(compressed_image_batch_name)
            # get the df columns that contains the compressed_image_batch_name
            columns_to_keep = [col for col in df.columns if compressed_image_batch_name in col]
            df_subset = df[columns_to_keep]
            for column in df_subset.columns:
                firing_rates = df_subset[column].loc[unit_id]
                #convert firing_rate str to list
                firing_rates = eval(firing_rates)
                max_firing_rate = max(firing_rates)
                firing_rates_list.append(max_firing_rate)
                image_index = int(column.split("--")[-1])
                sta_result.append(max_firing_rate/200 * compressed_image_data[image_index, :, :])
                X.append(compressed_image_data[image_index, :, :])
                Y.append(firing_rates)
        
        sta_result = np.array(sta_result)
        sta_result = sta_result.mean(axis = 0) - np.mean(firing_rates_list)/200 * all_images_mean
        print(sta_result.shape, unit_id)
        axs[k//col_num, k%col_num].imshow(sta_result, cmap = "jet")
        axs[k//col_num, k%col_num].set_title(unit_id)
        sta_result = np.abs(sta_result - sta_result.mean())
        receptive_field_center = np.unravel_index(np.argmax(sta_result), sta_result.shape)
        receptive_field_dict[unit_id] = receptive_field_center
        final_Y_dict[unit_id] = Y
    final_X_dict["X"] = X
    
    with open("final_X_dict.pkl", "wb") as f:
        pickle.dump(final_X_dict, f)
    with open("final_Y_dict.pkl", "wb") as f:
        pickle.dump(final_Y_dict, f)

        
    plt.show()
    with open("receptive_field_dict.pkl", "wb") as f:
        pickle.dump(receptive_field_dict, f)
    return receptive_field_dict

if __name__ == "__main__":
    # #########generate compressed image data
    # npy_file_name_list = glob.glob("ILSVRC2012_003R/*.npy")
    # npy_file_name_list.sort()
    # print(npy_file_name_list)
    # for npy_file_name in npy_file_name_list:    
    #     compressed_image_data = compress_image_data(npy_file_name,
    #                                                 show_plot = True,
    #                                                 save_compressed_image_data = True,
    #                                                 compressed_image_folder_name = "compressed_ILSVRC2012")
    
    
    # ####### check the compressed image data
    # compressed_image_data_list = glob.glob("compressed_ILSVRC2012/*.npy")
    # compressed_image_data_list.sort()
    # image_data_list = glob.glob("ILSVRC2012_npy/*.npy")
    # image_data_list.sort()
    # for compressed_image_data_name, image_data_name in zip(compressed_image_data_list, image_data_list):
    #     compressed_image_data = np.load(compressed_image_data_name)
    #     image_data = np.load(image_data_name)
    #     print(compressed_image_data_name, compressed_image_data.shape)
    #     print(image_data_name, image_data.shape)
    #     print(image_data_name, image_data.shape[0]/60)

    # ######### add connected image to firing time to pkl
    # compressed_image_data_name_list = glob.glob("compressed_ILSVRC2012/*.npy")
    # compressed_image_data_name_list.sort()
    # image_data_name_list = glob.glob("ILSVRC2012_003R/*.npy")
    # image_data_name_list.sort()
    # add_connected_image_to_firing_time_to_pkl("raw_data/2025.03.19-16.26.24-Rec.pkl",
    #                                           compressed_image_data_name_list,
    #                                           image_data_name_list,
    #                                           pre_start_frame_num = 181, #184
    #                                           inter_set_frame_num_list = [362, 361, 363],
    #                                           acq_rate = 20_000)
        
    # ####### validate the connected image to firing time
    # validate_connected_image_to_firing_time("raw_data/2025_connected_image_to_firing_time.pkl")
    # validate_connection_by_light_ref_plot("raw_data/2025_connected_image_to_firing_time.pkl")
    
    #make_simple_dataset("raw_data/2025_connected_image_to_firing_time.pkl")
    find_cell_center("2025_connected_image_to_firing_time.csv")