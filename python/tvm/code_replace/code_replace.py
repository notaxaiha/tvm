import json
import pdb


def code_replace(code, compute_dict, schedule_dict):
    return code

def code_replace_with_log_file(code, log_path):
    with open(log_path) as log_file:
        log_data = json.load(log_file)
        print("====log_data====")
        print(log_data)
        print("")

        log_input = log_data["input"]
        tensor_shapes = log_input[2]
        compute_dict = dict()

        featuremap_shape = tensor_shapes[0][1]
        compute_dict["in_size"] = featuremap_shape[0]
        compute_dict["batch"] = featuremap_shape[2]
        compute_dict["in_channel"] = featuremap_shape[3]

        conv_kernel_shape = tensor_shapes[1][1]
        compute_dict["kernel"] = featuremap_shape[0]
        compute_dict["num_filter"] = featuremap_shape[2]

        compute_dict["stride"] = tensor_shapes[2]
        compute_dict["padding"] = tensor_shapes[3]
        compute_dict["dilation"] = tensor_shapes[4]
        print("====compute_dict====")
        print(compute_dict)
        print("")

        log_config = log_data["config"]
        schedule_list = log_config["entity"]
        knob_list = [elem[0] for elem in schedule_list]
        value_list = [elem[2] for elem in schedule_list]
        schedule_dict = dict(zip(knob_list, value_list))
        print("====schedule_dict====")
        print(schedule_dict)
        print("")

        return code_replace(code, compute_dict, schedule_dict)

def code_replace_with_TVM_config(code, compute_dict):
    print("====compute_dict====")
    print(compute_dict)
    print("")
    
    
    schedule_dict = dict()
    print("====schedule_dict====")
    print(schedule_dict)
    print("")
    return code_replace(code, compute_dict, schedule_dict)
