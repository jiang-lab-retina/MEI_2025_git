import numpy as np

def convert_timestamp_to_frame_num(timestamps, frame_time):
    frame_num = [np.argmin(np.abs(frame_time - time)) for time in timestamps]
    return np.array(frame_num)   #return frame numbers in relative to the start of the frame_time
                     
                     
def get_sta(trace_to_analyze, movie_data_array, cover_range= (-60, 0), trace_type = "full"):
    sta_list = []  
    cover_length = cover_range[1] - cover_range[0]

    if cover_length < 0:
        raise ValueError("cover_range should be a tuple of (start, end) where start < end")
    # if cover_range[0] > 0:
    #     raise ValueError("cover_range[0] should be negative")
    if trace_type not in ["full", "timestamp"]:
        raise ValueError("trace_type should be 'full' or 'timestamp', timestamp means the trace_to_analyze is a list of timestamps")

    if trace_type == "full":
        for i in range(0, trace_to_analyze.shape[0] - cover_range[1]):
            if i + cover_range[0] >= 0 and i + cover_range[1] < movie_data_array.shape[0]:
                sta_iter = movie_data_array[i+cover_range[0]:i+cover_range[1]] * np.uint16(trace_to_analyze[i])
                sta_list.append(sta_iter)
                
        sta_list = np.array(sta_list)

    elif trace_type == "timestamp":
        for i in trace_to_analyze:
            if i + cover_range[0] >= 0 and i + cover_range[1] < movie_data_array.shape[0]:
                sta_list.append(movie_data_array[i+cover_range[0]:i+cover_range[1]])
        sta_list = np.array(sta_list)

    sta= sta_list.mean(axis=0)
    return sta