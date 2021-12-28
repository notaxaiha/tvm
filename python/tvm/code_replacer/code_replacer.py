import json
import pdb


class Code_replacer:
    enabled = False
    dump_path = None

    def __init__(self):
        self.compute_dict = dict()
        self.schedule_dict = dict()
        self.codegen_dict = dict()
        self.dump_count = 0
        self.code = None
        self.dumpcode = None

        self.wmma_m = 8
        self.wmma_n = 8
        self.wmma_k = 32

    def dump(self):
        filename = Code_replacer.dump_path + f"replacer_dump_{self.dump_count}.cu"
        self.dump_count += 1
        with open(filename, "w+") as f:
            f.write(self.dumpcode)
        return

    def generate_codegen_dict(self):
        in_size = self.compute_dict["in_size"]
        batch = self.compute_dict["batch"]
        in_channel = self.compute_dict["in_channel"]
        kernel = self.compute_dict["kernel"]
        num_filter = self.compute_dict["num_filter"]
        padding = self.compute_dict["padding"]

        block_row_warps = self.schedule_dict["block_row_warps"]
        block_col_warps = self.schedule_dict["block_col_warps"]
        warp_row_tiles = self.schedule_dict["warp_row_tiles"]
        warp_col_tiles = self.schedule_dict["warp_col_tiles"]
        chunk = self.schedule_dict["chunk"]


        row_tiles = batch * in_size * in_size // self.wmma_m
        warp_row_tiles_stable = min(warp_row_tiles, row_tiles)
        row_warps = row_tiles // warp_row_tiles_stable
        block_row_warps_stable = min(block_row_warps, row_warps)
        row_blocks = row_warps // block_row_warps_stable

        col_tiles = num_filter // self.wmma_n
        warp_col_tiles_stable = min(warp_col_tiles, col_tiles)
        col_warps = col_tiles // warp_col_tiles_stable
        block_col_warps_stable = min(block_col_warps, col_warps)
        col_blocks = col_warps // block_col_warps_stable

        chunk_stable = min(chunk, in_channel)

        warp_row_tiles = warp_row_tiles_stable
        block_row_warps = block_row_warps_stable
        warp_col_tiles = warp_col_tiles_stable
        block_col_warps = block_col_warps_stable
        chunk = chunk_stable

        codegen_dict = dict()
        codegen_dict["grid_row_blocks"] = row_blocks
        codegen_dict["block_row_warps"] = block_row_warps
        codegen_dict["warp_row_tiles"] = warp_row_tiles
        codegen_dict["grid_col_blocks"] = col_blocks
        codegen_dict["block_col_warps"] = block_col_warps
        codegen_dict["warp_col_tiles"] = warp_col_tiles

        codegen_dict["TB_ROW_COVER"] = block_row_warps * warp_row_tiles
        codegen_dict["WARP_ROW_COVER"] = warp_row_tiles
        codegen_dict["M_TB"] = block_row_warps * warp_row_tiles * self.wmma_m
        codegen_dict["N_TB"] = block_col_warps * warp_col_tiles * self.wmma_n
        codegen_dict["K_TB"] = in_channel
        codegen_dict["M_WARP"] = warp_row_tiles * self.wmma_m
        codegen_dict["N_WARP"] = warp_col_tiles * self.wmma_n
        codegen_dict["K_WARP"] = in_channel
        codegen_dict["IC_OUTER"] = chunk 
        codegen_dict["FEATUREMAP_SIZE"] = in_size
        codegen_dict["KERNEL_SIZE"] = kernel
        codegen_dict["PADDED_SIZE"] = in_size + 2 * padding
        codegen_dict["NUM_IC"] = in_channel

        return codegen_dict

    def codegen(self, kernel_intro):
        grid_row_blocks = self.codegen_dict["grid_row_blocks"] 
        block_row_warps = self.codegen_dict["block_row_warps"] 
        warp_row_tiles = self.codegen_dict["warp_row_tiles"] 
        grid_col_blocks = self.codegen_dict["grid_col_blocks"] 
        block_col_warps = self.codegen_dict["block_col_warps"] 
        warp_col_tiles = self.codegen_dict["warp_col_tiles"] 

        TB_ROW_COVER = self.codegen_dict["TB_ROW_COVER"] 
        WARP_ROW_COVER = self.codegen_dict["WARP_ROW_COVER"] 
        M_TB = self.codegen_dict["M_TB"] 
        N_TB = self.codegen_dict["N_TB"] 
        K_TB = self.codegen_dict["K_TB"] 
        M_WARP = self.codegen_dict["M_WARP"] 
        N_WARP = self.codegen_dict["N_WARP"] 
        K_WARP = self.codegen_dict["K_WARP"] 
        IC_OUTER = self.codegen_dict["IC_OUTER"] 
        FEATUREMAP_SIZE = self.codegen_dict["FEATUREMAP_SIZE"] 
        KERNEL_SIZE = self.codegen_dict["KERNEL_SIZE"] 
        PADDED_SIZE = self.codegen_dict["PADDED_SIZE"] 
        NUM_IC = self.codegen_dict["NUM_IC"] 
        PACK_RATE = 8


        def add_codeline(codelist, codeline, indent_level=1):
            indent = "\t" * indent_level
            codelist.append(indent + codeline + "\n")

        result_codelist = []
        add_codeline(result_codelist, kernel_intro[0], 0)

        #initial declaration & initialization
        accum_fragments = (M_WARP * N_WARP) // (self.wmma_m * self.wmma_n)
        featuremap_shared_size = (TB_ROW_COVER + 2) * K_TB
        featuremap_fragments = (M_WARP * K_WARP) // (self.wmma_m * self.wmma_k * IC_OUTER)
        kernel_shared_size = max((K_TB * N_TB) // (PACK_RATE * IC_OUTER), (M_TB * N_TB) / PACK_RATE)
        kernel_fragments = (K_WARP * N_WARP) // (self.wmma_k * self.wmma_n * IC_OUTER)

        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::accumulator, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, int> Conv_wmma_accumulator[{accum_fragments}];")
        add_codeline(result_codelist, f"__shared__ int featuremap_shared[{featuremap_shared_size}];")
        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::row_major> featuremap_frag[{featuremap_fragments}];")
        add_codeline(result_codelist, f"__shared__ int kernel_shared[{kernel_shared_size}];")
        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> kernel_frag[{kernel_fragments}];")
        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int o_c_init = 0; o_c_init < accum_fragments; ++o_c_init) {{")
        add_codeline(result_codelist, f"(void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);", 2)
        add_codeline(result_codelist, f"}}")
        add_codeline(result_codelist, f"int outfeature_row = (blockIdx.z * {TB_ROW_COVER}) / {FEATUREMAP_SIZE};")
        add_codeline(result_codelist, f"int outfeature_col = (blockIdx.z * {TB_ROW_COVER}) % {FEATUREMAP_SIZE};")

        #featuremap load to shared memory
        featuremap_shared_size_vectorized = featuremap_shared_size // 4
        load_iteration = featuremap_shared_size_vectorized // (block_row_warps * block_col_warps * 32)



        add_codeline(result_codelist, f"for (int kh = 0; kh < 3; kh++) {{")
        add_codeline(result_codelist, f"int cur_row = outfeature_row + kh;", 2)
        add_codeline(result_codelist, f"int base_addr = cur_row * PADDED_SIZE * NUM_IC + outfeature_col * NUM_IC;", 2)
        add_codeline(result_codelist, f"#pragma unroll", 2)
        add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iteration}; load_iter++) {{", 2)
        
        add_codeline(result_codelist, f"}}", 0)

        result_code = "".join(result_codelist)

        return result_code

    def code_replace(self):
        in_code = self.code
        kernel_intro = []

        self.codegen_dict = self.generate_codegen_dict()
        print("====codegen_dict====")
        print(self.codegen_dict)
        print("")

        for line in in_code.splitlines():
            if "__launch_bounds__" in line:
                kernel_intro.append(line)

        #print(kernel_intro)

        result_code = self.codegen(kernel_intro)
        self.dumpcode = result_code

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
