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

        chunk_stable = min(chunk, in_channel//self.wmma_k)

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
        codegen_dict["IC_OUTER"] = (in_channel // self.wmma_k) // chunk
        codegen_dict["IC_INNER"] = chunk
        codegen_dict["FEATUREMAP_SIZE"] = in_size
        codegen_dict["KERNEL_SIZE"] = kernel
        codegen_dict["PADDED_SIZE"] = in_size + 2 * padding
        codegen_dict["NUM_IC"] = in_channel
        codegen_dict["NUM_OC"] = num_filter

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
        IC_INNER = self.codegen_dict["IC_INNER"] 
        FEATUREMAP_SIZE = self.codegen_dict["FEATUREMAP_SIZE"] 
        KERNEL_SIZE = self.codegen_dict["KERNEL_SIZE"] 
        PADDED_SIZE = self.codegen_dict["PADDED_SIZE"] 
        NUM_IC = self.codegen_dict["NUM_IC"] 
        NUM_OC = self.codegen_dict["NUM_OC"] 
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
        kernel_shared_size = max((K_TB * N_TB) // (PACK_RATE * IC_OUTER), (M_TB * N_TB) // PACK_RATE)
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
        add_codeline(result_codelist, f"int outfeature_row = (blockIdx.x * {TB_ROW_COVER}) / {FEATUREMAP_SIZE};")
        add_codeline(result_codelist, f"int outfeature_col = (blockIdx.x * {TB_ROW_COVER}) % {FEATUREMAP_SIZE};")


        #main loop
        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int kh = 0; kh < KERNEL_SIZE; kh++) {{")

        #featuremap load to shared memory
        add_codeline(result_codelist, f"int cur_row = outfeature_row + kh;", 2)
        add_codeline(result_codelist, f"int base_addr = cur_row * PADDED_SIZE * NUM_IC + outfeature_col * NUM_IC;", 2)


        featuremap_shared_size_vectorized = featuremap_shared_size // 4
        whole_block_load_size = (block_row_warps * block_col_warps * 32)
        load_iteration = featuremap_shared_size_vectorized // whole_block_load_size
        if load_iteration != 0:
            add_codeline(result_codelist, f"#pragma unroll", 2)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iteration}; load_iter++) {{", 2)
            add_codeline(result_codelist, f"int featuremap_addr = (base_addr/4) + (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[(load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)


        featuremap_rest = featuremap_shared_size_vectorized % whole_block_load_size
        inner_block_load_size = (block_row_warps * 32)
        load_parallel_1 = featuremap_rest // inner_block_load_size
        if load_parallel_1 != 0:
            add_codeline(result_codelist, f"if(threadIdx.z < {load_parallel_1}) {{",2)
            add_codeline(result_codelist, f"int featuremap_addr = (base_addr/4) + ({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)


        featuremap_rest = featuremap_rest % inner_block_load_size
        whole_warp_load_size = 32
        load_parallel_2 = featuremap_rest // whole_warp_load_size
        if load_parallel_2 != 0:
            add_codeline(result_codelist, f"if(threadIdx.y < {load_parallel_2}) {{",2)
            add_codeline(result_codelist, f"int featuremap_addr = (base_addr/4) + ({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_parallel_1} * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_paralel_1} * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)

        add_codeline(result_codelist, f"__syncthreads();",2)

        #second loop
        add_codeline(result_codelist, f"#pragma unroll",2)
        add_codeline(result_codelist, f"for (int ic_outer = 0; ic_outer < {IC_OUTER}; ic_outer++) {{",2)
        add_codeline(result_codelist, f"#pragma unroll",3)
        add_codeline(result_codelist, f"for (int kw = 0; kw < KERNEL_SIZE; kw++) {{",3)
        add_codeline(result_codelist, f"__syncthreads();",3)

        #load featuremap to fragments
        add_codeline(result_codelist, f"if (kw==0) {{",3)
        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for(int row_iter = 0; row_iter < {WARP_ROW_COVER}; row_iter++) {{",4)
        add_codeline(result_codelist, f"int shared_mem_idx = (threadIdx.y * {WARP_ROW_COVER} + row_iter);",5)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",5)
        add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[row_iter * {IC_INNER} + ic_inner], ((int *)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {self.wmma_k}) + (ic_outer * {IC_INNER} * {self.wmma_k}) + (ic_inner * {self.wmma_k})), 32);", 6)
        add_codeline(result_codelist, f"}}",4)
        add_codeline(result_codelist, f"}}",3)

        add_codeline(result_codelist, f"else {{",3)
        add_codeline(result_codelist, f"int shared_mem_idx = ((threadIdx.y + 1) * WARP_ROW_COVER + kw - 1);",4)
        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",4)
        add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[(kw - 1) * {IC_INNER} + ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {self.wmma_k}) + (ic_outer * {IC_INNER} * {self.wmma_k}) + (ic_inner * {self.wmma_k})), 32);",5)
        add_codeline(result_codelist, f"}}",3)
        add_codeline(result_codelist, f"__syncthreads();",3)

        #load weight to shared memory
        #should find where ic_outer is on memory space
        #weight layout = HWOIoi, H_W_{O_blocks}_{O_warps}_{O_tiles}_{IC_OUTER}_{IC_INNER}_{wmma_n}_{wmma_k}
        assert(kernel_shared_size > whole_block_load_size)


        ic_outer_multiplier = IC_INNER * self.wmma_n * self.wmma_k // PACK_RATE
        location = 0 if ic_outer_multiplier >= whole_block_load_size else (1 if ic_outer_multiplier >= inner_block_load_size else 2)
        print(f"location:{location}")
        print(f"ic_outer_multiplier:{ic_outer_multiplier}")


        add_codeline(result_codelist, f"int base_addr = (kh * {KERNEL_SIZE} * {NUM_OC} * {NUM_IC} / {PACK_RATE}) + (kw * {NUM_OC} * {NUM_IC} / {PACK_RATE}) + (blockIdx.y * {N_TB} * {NUM_IC} / {PACK_RATE});", 3)
        #ic_outer cover is bigger than the whole block load
        if location == 0:
            load_iter = kernel_shared_size // whole_block_load_size
            load_iter_inner = ic_outer_multiplier // whole_block_load_size
            load_iter_outer = load_iter // load_iter_inner
            add_codeline(result_codelist, f"#pragma unroll",3)
            add_codeline(result_codelist, f"for (int load_iter_outer = 0; load_iter_outer < {load_iter_outer}; load_iter_outer++) {{",3)
            add_codeline(result_codelist, f"#pragma unroll",4)
            add_codeline(result_codelist, f"for (int load_iter_inner = 0; load_iter_inner < {load_iter_inner}; load_iter_inner++) {{",4)
            add_codeline(result_codelist, f"int kernel_global_src = (load_iter_outer * {IC_OUTER} * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (ic_outer * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (load_iter_inner * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter_outer * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (load_iter_inner * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",5)
            add_codeline(result_codelist, f"}}",4)
            add_codeline(result_codelist, f"}}",3)

        #ic_outer cover is bigger than the inner block load
        if location == 1:
            load_iter = kernel_shared_size // whole_block_load_size
            #bits = block_col_warps.bit_length() - 1

            load_parallel_inner = ic_outer_multiplier // inner_block_load_size
            assert(load_parallel_inner & (load_parallel_inner-1) == 0)
            load_parallel_inner_bitmask = load_parallel_inner - 1

            load_parallel_outer = block_col_warps // load_parallel_inner
            assert(load_parallel_outer & (load_parallel_outer-1) == 0)
            load_parallel_outer_bitmask = block_col_warps - load_parallel_outer

            add_codeline(result_codelist, f"#pragma unroll",3)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{",3)
            add_codeline(result_codelist, f"int kernel_global_src = (load_iter * {IC_OUTER} * {block_col_warps} * {block_row_warps} * 32) + ((threadIdx.z & {load_parallel_outer_bitmask}) * {IC_OUTER} * {load_parallel_inner} * {block_row_warps} * 32) + (ic_outer * {load_parallel_inner} * {block_row_warps} * 32) + ((threadIdx.z & {load_parallel_inner_bitmask}) * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",4)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",4)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",4)
            add_codeline(result_codelist, f"}}",3)

        #ic_outer cover is smaller than the inner block load
        if location == 2:
            load_iter = kernel_shared_size // whole_block_load_size
            #bits = block_col_warps.bit_length() - 1

            load_parallel_inner = ic_outer_multiplier // whole_warp_load_size
            assert(load_parallel_inner & (load_parallel_inner-1) == 0)
            load_parallel_inner_bitmask = load_parallel_inner - 1

            load_parallel_outer = block_col_warps // load_parallel_inner
            assert(load_parallel_outer & (load_parallel_outer-1) == 0)
            load_parallel_outer_bitmask = block_col_warps - load_parallel_outer

            add_codeline(result_codelist, f"#pragma unroll",3)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{",3)
            add_codeline(result_codelist, f"int kernel_global_src = (load_iter * {block_col_warps} * {IC_OUTER} * {block_row_warps} * 32) + (threadIdx.z * {IC_OUTER} * {block_row_warps} * 32) + ((threadIdx.y & {load_parallel_outer_bitmask}) * {IC_OUTER} * {load_parallel_inner} * 32) + (ic_outer * {load_parallel_inner} * 32) + ((threadIdx.y & {load_parallel_inner_bitmask}) * 32) + (threadIdx.x);",4)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",4)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",4)
            add_codeline(result_codelist, f"}}",3)

        #load to fragment
        
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
