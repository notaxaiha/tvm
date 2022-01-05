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




        ################################################
        ################################################
        ##### Initial declaration & initialization #####
        ################################################
        ################################################

        accum_fragments = (M_WARP * N_WARP) // (self.wmma_m * self.wmma_n)
        featuremap_shared_size = (TB_ROW_COVER + KERNEL_SIZE - 1) * K_TB
        featuremap_fragments = (M_WARP * K_WARP) // (self.wmma_m * self.wmma_k * IC_OUTER)
        kernel_shared_size = max((K_TB * N_TB) // (PACK_RATE * IC_OUTER), (M_TB * N_TB) // PACK_RATE)
        kernel_fragments = (K_WARP * N_WARP) // (self.wmma_k * self.wmma_n * IC_OUTER)

        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::accumulator, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, int> Conv_wmma_accumulator[{accum_fragments}];")
        add_codeline(result_codelist, f"__shared__ int featuremap_shared[{featuremap_shared_size}];")
        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::row_major> featuremap_frag[{featuremap_fragments}];")
        ########using u4 on featuremap for correctness check
        #add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> featuremap_frag[{featuremap_fragments}];")
        add_codeline(result_codelist, f"__shared__ int kernel_shared[{kernel_shared_size}];")
        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> kernel_frag[{kernel_fragments}];")
        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int o_c_init = 0; o_c_init < {accum_fragments}; ++o_c_init) {{")
        add_codeline(result_codelist, f"(void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);", 2)
        add_codeline(result_codelist, f"}}")
        add_codeline(result_codelist, f"int outfeature_row = (blockIdx.x * {TB_ROW_COVER}) / {FEATUREMAP_SIZE};")
        add_codeline(result_codelist, f"int outfeature_col = (blockIdx.x * {TB_ROW_COVER}) % {FEATUREMAP_SIZE};")

        ################################################
        ################################################
        ################## Main loop ###################
        ################################################
        ################################################

        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int kh = 0; kh < {KERNEL_SIZE}; kh++) {{")

        ################################################
        ################################################
        ###### Featuremap load to shared memory ########
        ################################################
        ################################################

        add_codeline(result_codelist, f"int cur_row = outfeature_row + kh;", 2)
        add_codeline(result_codelist, f"int featuremap_base_addr = cur_row * {PADDED_SIZE} * {NUM_IC} + outfeature_col * {NUM_IC};", 2)


        featuremap_shared_size_vectorized = featuremap_shared_size // 4
        whole_block_load_size = (block_row_warps * block_col_warps * 32)
        load_iteration = featuremap_shared_size_vectorized // whole_block_load_size
        if load_iteration != 0:
            add_codeline(result_codelist, f"#pragma unroll", 2)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iteration}; load_iter++) {{", 2)
            add_codeline(result_codelist, f"int featuremap_addr = (featuremap_base_addr/4) + (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[(load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)


        featuremap_rest = featuremap_shared_size_vectorized % whole_block_load_size
        inner_block_load_size = (block_row_warps * 32)
        load_parallel_1 = featuremap_rest // inner_block_load_size
        if load_parallel_1 != 0:
            add_codeline(result_codelist, f"if(threadIdx.z < {load_parallel_1}) {{",2)
            add_codeline(result_codelist, f"int featuremap_addr = (featuremap_base_addr/4) + ({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)


        featuremap_rest = featuremap_rest % inner_block_load_size
        whole_warp_load_size = 32
        load_parallel_2 = featuremap_rest // whole_warp_load_size
        if load_parallel_2 != 0:
            add_codeline(result_codelist, f"if((threadIdx.y < {load_parallel_2}) && (threadIdx.z == {block_col_warps} - 1)) {{",2)
            add_codeline(result_codelist, f"int featuremap_addr = (featuremap_base_addr/4) + ({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_parallel_1} * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_parallel_1} * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)

        featuremap_rest = featuremap_rest % whole_warp_load_size
        load_parallel_3 = featuremap_rest
        if load_parallel_3 != 0:
            add_codeline(result_codelist, f"if((threadIdx.x < {load_parallel_3}) && (threadIdx.y == {block_row_warps} - 1) && (threadIdx.z == {block_col_warps} - 1)) {{",2)
            add_codeline(result_codelist, f"int featuremap_addr = (featuremap_base_addr/4) + ({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_parallel_1} * {block_row_warps} * 32) + ({load_parallel_2} * 32) + (threadIdx.x);",3)
            add_codeline(result_codelist, f"((int4*)featuremap_shared)[({load_iteration} * {block_col_warps} * {block_row_warps} * 32) + ({load_parallel_1} * {block_row_warps} * 32) + ({load_parallel_2} * 32) + (threadIdx.x)] = ((int4*)featuremap)[featuremap_addr];",3)
            add_codeline(result_codelist, f"}}",2)

        add_codeline(result_codelist, f"__syncthreads();",2)

        ################################################
        ################################################
        ################# Second loop ##################
        ################################################
        ################################################

        add_codeline(result_codelist, f"#pragma unroll",2)
        add_codeline(result_codelist, f"for (int ic_outer = 0; ic_outer < {IC_OUTER}; ic_outer++) {{",2)
        add_codeline(result_codelist, f"#pragma unroll",3)
        add_codeline(result_codelist, f"for (int kw = 0; kw < {KERNEL_SIZE}; kw++) {{",3)
        add_codeline(result_codelist, f"__syncthreads();",4)


        ################################################
        ################################################
        ######## Load featuremap to fragments ##########
        ################################################
        ################################################

        fragment_size = self.wmma_m * self.wmma_k // PACK_RATE
        add_codeline(result_codelist, f"if (kw==0) {{",4)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for(int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",5)
        add_codeline(result_codelist, f"int shared_mem_idx = (threadIdx.y * {warp_row_tiles} + row_iter);",6)
        add_codeline(result_codelist, f"#pragma unroll",6)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",6)
        add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[row_iter * {IC_INNER} + ic_inner], ((int *)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);", 7)
        add_codeline(result_codelist, f"}}",5)
        add_codeline(result_codelist, f"}}",4)

        add_codeline(result_codelist, f"else {{",4)
        add_codeline(result_codelist, f"int shared_mem_idx = ((threadIdx.y + 1) * {warp_row_tiles} + kw - 1);",5)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",5)
        if warp_row_tiles != 1:
            add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[(kw - 1) * {IC_INNER} + ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);",6)
        else:
            add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);",6)

        add_codeline(result_codelist, f"}}",4)
        add_codeline(result_codelist, f"__syncthreads();",4)




        ################################################
        ################################################
        ######## Load weight to shared memory ##########
        ################################################
        ################################################

        #should find where ic_outer is on memory space
        #weight layout = HWOIoi, H_W_{O_blocks}_{O_warps}_{O_tiles}_{IC_OUTER}_{IC_INNER}_{wmma_n}_{wmma_k}

        assert(kernel_shared_size > whole_block_load_size)


        ic_outer_multiplier = IC_INNER * self.wmma_n * self.wmma_k // PACK_RATE
        location = 0 if ic_outer_multiplier >= whole_block_load_size else (1 if ic_outer_multiplier >= inner_block_load_size else 2)
        print(f"location:{location}")
        print(f"ic_outer_multiplier:{ic_outer_multiplier}")


        add_codeline(result_codelist, f"int kernel_base_addr = (kh * {KERNEL_SIZE} * {NUM_OC} * {NUM_IC} / {PACK_RATE}) + (kw * {NUM_OC} * {NUM_IC} / {PACK_RATE}) + (blockIdx.y * {N_TB} * {NUM_IC} / {PACK_RATE});", 4)
        #ic_outer cover is bigger than the whole block load
        if location == 0:
            load_iter = kernel_shared_size // whole_block_load_size
            load_iter_inner = ic_outer_multiplier // whole_block_load_size
            load_iter_outer = load_iter // load_iter_inner
            add_codeline(result_codelist, f"#pragma unroll",4)
            add_codeline(result_codelist, f"for (int load_iter_outer = 0; load_iter_outer < {load_iter_outer}; load_iter_outer++) {{",4)
            add_codeline(result_codelist, f"#pragma unroll",5)
            add_codeline(result_codelist, f"for (int load_iter_inner = 0; load_iter_inner < {load_iter_inner}; load_iter_inner++) {{",5)
            add_codeline(result_codelist, f"int kernel_global_src = kernel_base_addr + (load_iter_outer * {IC_OUTER} * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (ic_outer * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (load_iter_inner * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",6)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter_outer * {load_iter_inner} * {block_col_warps} * {block_row_warps} * 32) + (load_iter_inner * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",6)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",6)
            add_codeline(result_codelist, f"}}",5)
            add_codeline(result_codelist, f"}}",4)

        #ic_outer cover is bigger than the inner block load
        if location == 1:
            load_iter = kernel_shared_size // whole_block_load_size
            #bits = block_col_warps.bit_length() - 1

            #1,2,4,8.....
            load_parallel = ic_outer_multiplier // inner_block_load_size
            assert(load_parallel & (load_parallel-1) == 0)
            load_parallel_bitmask = load_parallel - 1

            denom = load_parallel
            load_parallel_power = 0
            while(denom != 1):
                denom /= 2
                load_parallel_power += 1

            add_codeline(result_codelist, f"#pragma unroll",4)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{",4)
            add_codeline(result_codelist, f"int kernel_global_src = kernel_base_addr + (load_iter * {IC_OUTER} * {block_col_warps} * {block_row_warps} * 32) + ((threadIdx.z >> {load_parallel_power}) * {IC_OUTER} * {load_parallel} * {block_row_warps} * 32) + (ic_outer * {load_parallel} * {block_row_warps} * 32) + ((threadIdx.z & {load_parallel_bitmask}) * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",5)
            add_codeline(result_codelist, f"}}",4)

        #ic_outer cover is smaller than the inner block load
        if location == 2:
            load_iter = kernel_shared_size // whole_block_load_size
            #bits = block_col_warps.bit_length() - 1

            load_parallel = ic_outer_multiplier // whole_warp_load_size
            assert(load_parallel & (load_parallel-1) == 0)
            load_parallel_bitmask = load_parallel - 1

            denom = load_parallel
            load_parallel_power = 0
            while(denom != 1):
                denom /= 2
                load_parallel_power += 1

            add_codeline(result_codelist, f"#pragma unroll",4)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{",4)
            add_codeline(result_codelist, f"int kernel_global_src = kernel_base_addr + (load_iter * {block_col_warps} * {IC_OUTER} * {block_row_warps} * 32) + (threadIdx.z * {IC_OUTER} * {block_row_warps} * 32) + ((threadIdx.y >> {load_parallel_power}) * {IC_OUTER} * {load_parallel} * 32) + (ic_outer * {load_parallel} * 32) + ((threadIdx.y & {load_parallel_bitmask}) * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"int kernel_shared_dst = (load_iter * {block_col_warps} * {block_row_warps} * 32) + (threadIdx.z * {block_row_warps} * 32) + (threadIdx.y * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"((int*)kernel_shared)[kernel_shared_dst] = ((int*)kernel)[kernel_global_src];",5)
            add_codeline(result_codelist, f"}}",4)

        add_codeline(result_codelist, f"__syncthreads();",4)



        ################################################
        ################################################
        ######## Load kernel weight to fragments #######
        ################################################
        ################################################

        fragment_size = self.wmma_k * self.wmma_n // PACK_RATE
        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for (int oc_tile = 0; oc_tile < {warp_col_tiles}; oc_tile++) {{",4)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for (int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++) {{",5)
        add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(kernel_frag[oc_tile * {IC_INNER} + ic_inner], ((int*)kernel_shared + (threadIdx.z * {warp_col_tiles} * {IC_INNER} * {fragment_size}) + (oc_tile * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);",6)
        add_codeline(result_codelist, f"}}",5)
        add_codeline(result_codelist, f"}}",4)


        ################################################
        ################################################
        ################# Conduct wmma #################
        ################################################
        ################################################

        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for (int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++) {{",4)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for (int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",5)
        add_codeline(result_codelist, f"int featuremap_fragment_idx = ((row_iter + kw) % {warp_row_tiles}) * {IC_INNER} + ic_inner;",6)
        add_codeline(result_codelist, f"#pragma unroll",6)
        add_codeline(result_codelist, f"for (int oc_tile = 0; oc_tile < {warp_col_tiles}; oc_tile++) {{",6)
        add_codeline(result_codelist, f"(void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[row_iter * {warp_col_tiles} + oc_tile], featuremap_frag[featuremap_fragment_idx], kernel_frag[oc_tile * {IC_INNER} + ic_inner], Conv_wmma_accumulator[row_iter * {warp_col_tiles} + oc_tile]);",7)
        add_codeline(result_codelist, f"}}",6)
        add_codeline(result_codelist, f"}}",5)
        add_codeline(result_codelist, f"}}",4)
        add_codeline(result_codelist, f"}}",3)
        add_codeline(result_codelist, f"}}",2)
        add_codeline(result_codelist, f"}}",1)
        add_codeline(result_codelist, f"__syncthreads();",1)


        ################################################
        ################################################
        ##### Conduct fragment level data packing ######
        ################################################
        ################################################
        #each warp should cover at least four fragment tiles on output channel level for warp-level register packing
        warp_register_count = 32
        packed_tile_register_count = self.wmma_m * self.wmma_n // PACK_RATE

        tiles_for_packing = warp_register_count // packed_tile_register_count

        assert(warp_col_tiles >= tiles_for_packing)
        packing_iter = warp_col_tiles // tiles_for_packing

        add_codeline(result_codelist, f"#pragma unroll",1)
        add_codeline(result_codelist, f"for (int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",1)
        add_codeline(result_codelist, f"#pragma unroll",2)
        add_codeline(result_codelist, f"for (int packing_iter = 0; packing_iter < {packing_iter}; packing_iter++) {{",2)
        add_codeline(result_codelist, f"int fully_packed = 0;",3)
        add_codeline(result_codelist, f"#pragma unroll",3)
        add_codeline(result_codelist, f"for (int output_tile_iter = 0; output_tile_iter < {tiles_for_packing}; output_tile_iter++) {{",3)
        add_codeline(result_codelist, f"int partial_packed = 0;",4)
        add_codeline(result_codelist, f"int temp;",4)
        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for (int elem_iter = 0; elem_iter < Conv_wmma_accumulator[0].num_elements; elem_iter++) {{",4)
        add_codeline(result_codelist, f"partial_packed <<= 4;", 5)
        add_codeline(result_codelist, f"int outval = Conv_wmma_accumulator[row_iter * {warp_col_tiles} + packing_iter * {tiles_for_packing} + output_tile_iter].x[elem_iter];", 5)
        #add_codeline(result_codelist, f"outval += ((int*)bias)[2*threadIdx.x + elem_iter];", 5)
        #this line is required for correctness check 
        #add_codeline(result_codelist, f"outval = min(((max(outval, 0) << (long)4) * (long)1241513984 + (long)1073741824) >> (long)31, 15);", 5)
        #this line is required for correctness check 
        add_codeline(result_codelist, f"outval &= 0xf;", 5)
        add_codeline(result_codelist, f"partial_packed |= outval;", 5)
        add_codeline(result_codelist, f"}}",4)
        add_codeline(result_codelist, f"partial_packed <<= 24;",4)
        add_codeline(result_codelist, f"temp = warpReducePack(partial_packed);",4)
        add_codeline(result_codelist, f"temp = __shfl_up_sync(0xffffffff, temp, output_tile_iter);",4)
        add_codeline(result_codelist, f"if(threadIdx.x == 0 && output_tile_iter > 0)",4)
        add_codeline(result_codelist, f"temp = 0;",5)
        add_codeline(result_codelist, f"fully_packed |= temp;",4)
        add_codeline(result_codelist, f"}}",3)
        #this shared memory address tends to shift due to data layout change
        add_codeline(result_codelist, f"kernel_shared[(threadIdx.y * {warp_row_tiles} * {block_col_warps} * {packing_iter} * 32) + (row_iter * {block_col_warps} * {packing_iter} * 32) + (threadIdx.z * {packing_iter} * 32) + (packing_iter * 32) + threadIdx.x] = fully_packed;",3)
        add_codeline(result_codelist, f"__syncthreads();",3)
        add_codeline(result_codelist, f"}}",2)
        add_codeline(result_codelist, f"__syncthreads();",2)
        add_codeline(result_codelist, f"}}",1)



        add_codeline(result_codelist, f"#pragma unroll",1)
        add_codeline(result_codelist, f"for (int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",1)
        add_codeline(result_codelist, f"#pragma unroll",2)
        #packing iter iterates over output channel
        channel_iter = packing_iter
        add_codeline(result_codelist, f"for (int channel_iter = 0; channel_iter < {channel_iter}; channel_iter++) {{",2)
        add_codeline(result_codelist, f"int global_output_dst = (blockIdx.x * {block_row_warps} * {warp_row_tiles} * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",3)
        add_codeline(result_codelist, f"+ (threadIdx.y * {warp_row_tiles} * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",5)
        add_codeline(result_codelist, f"+ (row_iter * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",5)
        add_codeline(result_codelist, f"+ (blockIdx.y * {block_col_warps} * {channel_iter} * 32) + (threadIdx.z * {channel_iter} * 32) + (channel_iter * 32) + (threadIdx.x);",5)
        add_codeline(result_codelist, f"int local_output_src = (threadIdx.y * {warp_row_tiles} * {block_col_warps} * {channel_iter} * 32)",3)
        add_codeline(result_codelist, f"+ (row_iter * {block_col_warps} * {channel_iter} * 32) + (threadIdx.z * {channel_iter} * 32) + (channel_iter * 32) + (threadIdx.x);",5)
        add_codeline(result_codelist, f"((int*)T_cast)[global_output_dst] = kernel_shared[local_output_src];", 3)
        
        add_codeline(result_codelist, f"}}", 2)
        add_codeline(result_codelist, f"}}", 1)
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
