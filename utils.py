import datetime


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def get_total_elements_cnt(x):
    tensor_shape = x.size()
    count = 1
    for dim_size in tensor_shape:
        count *= dim_size
    return count
