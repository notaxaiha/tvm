import json
import pdb


class Code_replacer:
    enabled = False
    dump_path = None

    def __init__(self):
        self.compute_dict = dict()
        self.schedule_dict = dict()
        self.dump_count = 0
        self.code = None

    def dump(self):
        filename = Code_replacer.dump_path + f"replacer_dump_{self.dump_count}.cu"
        self.dump_count += 1
        with open(filename, "w+") as f:
            f.write(self.code)
        return

    def code_replace(self):
        return self.code

    def code_replace_with_log_file(self, code, log_path):
        self.code = code
        self.code += "//logcall\n"
        with open(log_path) as log_file:
            log_data = json.load(log_file)
            print("====log_data====")
            print(log_data)
            print("")

            log_input = log_data["input"]
            tensor_shapes = log_input[2]

            featuremap_shape = tensor_shapes[0][1]
            self.compute_dict["in_size"] = featuremap_shape[0]
            self.compute_dict["batch"] = featuremap_shape[2]
            self.compute_dict["in_channel"] = featuremap_shape[3]

            conv_kernel_shape = tensor_shapes[1][1]
            self.compute_dict["kernel"] = conv_kernel_shape[0]
            self.compute_dict["num_filter"] = conv_kernel_shape[2]

            self.compute_dict["stride"] = tensor_shapes[2]
            self.compute_dict["padding"] = tensor_shapes[3]
            self.compute_dict["dilation"] = tensor_shapes[4]
            print("====compute_dict====")
            print(self.compute_dict)
            print("")

            log_config = log_data["config"]
            schedule_list = log_config["entity"]
            knob_list = [elem[0] for elem in schedule_list]
            value_list = [elem[2] for elem in schedule_list]
            self.schedule_dict = dict(zip(knob_list, value_list))
            print("====schedule_dict====")
            print(self.schedule_dict)
            print("")

            return self.code_replace()

    def code_replace_with_TVM_config(self, code):
        self.code = code
        self.code += "//configcall\n"
        print("====compute_dict====")
        print(self.compute_dict)
        print("")
        
        
        print("====schedule_dict====")
        print(self.schedule_dict)
        print("")
        return self.code_replace()
