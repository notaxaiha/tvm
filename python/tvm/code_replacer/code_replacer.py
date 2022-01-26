import json
import numpy
import pdb
import os

class Code_replacer:
    enabled = False
    dump_path = None

    def __init__(self):
        self.compute_dict = dict()
        self.schedule_dict = dict()
        self.codegen_dict = dict()
        self.dump_count = 0
        self.tune = False
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
        in_height = self.compute_dict["in_height"]
        in_width = self.compute_dict["in_width"]
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

        reorder_inner = self.schedule_dict["reorder_inner"]
        ko_kh_reorder = True if reorder_inner == [1,0] else False


        row_tiles = batch * in_height * in_width // self.wmma_m
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

        if not self.tune:
            print("============== Binding Info ===============")
            print("BlockIdx.x : ", row_blocks)
            print("BlockIdx.y : ", col_blocks)
            print("BlockIdx.z : ", 1)
            print("ThreadIdx.x : ", 32)
            print("ThreadIdx.y : ", block_row_warps)
            print("ThreadIdx.z : ", block_col_warps)
            print("===========================================")

        f = open("/tmp/index.txt", 'w')
        f.write("%d\n" %row_blocks)
        f.write("%d\n" %col_blocks)
        f.write("%d\n" %1)
        f.write("%d\n" %32)
        f.write("%d\n" %block_row_warps)
        f.write("%d\n" %block_col_warps)
        f.close()

        if((in_width * batch) % (block_row_warps * warp_row_tiles * self.wmma_m) != 0):
            codegen_dict["Fallback"] = True
            return codegen_dict

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
        codegen_dict["FEATUREMAP_WIDTH"] = in_width
        codegen_dict["FEATUREMAP_HEIGHT"] = in_height
        codegen_dict["KERNEL_SIZE"] = kernel
        codegen_dict["NUM_IC"] = in_channel
        codegen_dict["NUM_OC"] = num_filter
        codegen_dict["PADDING"] = padding
        codegen_dict["REORDER"] = ko_kh_reorder

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
        FEATUREMAP_WIDTH = self.codegen_dict["FEATUREMAP_WIDTH"] 
        FEATUREMAP_HEIGHT = self.codegen_dict["FEATUREMAP_HEIGHT"] 
        KERNEL_SIZE = self.codegen_dict["KERNEL_SIZE"] 
        NUM_IC = self.codegen_dict["NUM_IC"] 
        NUM_OC = self.codegen_dict["NUM_OC"] 
        PADDING = self.codegen_dict["PADDING"] 
        REORDER = self.codegen_dict["REORDER"] 
        PACK_RATE = 8

        manual_correctness_check = False


        def add_codeline(codelist, codeline, indent_level=1):
            indent = "\t" * indent_level
            codelist.append(indent + codeline + "\n")

        result_codelist = []
        
        add_codeline(result_codelist, f"__device__ inline long max(int a, long b)", 0)
        add_codeline(result_codelist, f"{{", 0)
        add_codeline(result_codelist, f"return max((long)a, b);", 1)
        add_codeline(result_codelist, f"}}", 0)
        add_codeline(result_codelist, f"__device__ inline long max(long a, int b)", 0)
        add_codeline(result_codelist, f"{{", 0)
        add_codeline(result_codelist, f"return max(b, a);", 1)
        add_codeline(result_codelist, f"}}", 0)
        add_codeline(result_codelist, f"__device__ inline long min(long a, int b)", 0)
        add_codeline(result_codelist, f"{{", 0)
        add_codeline(result_codelist, f"return min(a, (long)b);", 1)
        add_codeline(result_codelist, f"}}", 0)
        add_codeline(result_codelist, f"__device__ __inline__ int warpReducePack(unsigned int val) {{", 0)
        add_codeline(result_codelist, f"#pragma unroll", 1)
        add_codeline(result_codelist, f"for(int repeat=0;repeat<2;repeat++)", 1)
        add_codeline(result_codelist, f"{{", 1)
        add_codeline(result_codelist, f"unsigned temp = __shfl_down_sync(0xffffffff, val, repeat + 1, (repeat + 1)*2);", 2)
        add_codeline(result_codelist, f"val |= (temp >> (8* (repeat + 1)));", 2)
        add_codeline(result_codelist, f"}}", 1)
        add_codeline(result_codelist, f"if(threadIdx.x % 4 != 0)", 1)
        add_codeline(result_codelist, f"val = 0;", 2)
        add_codeline(result_codelist, f"return val;", 1)
        add_codeline(result_codelist, f"}}", 0)
        add_codeline(result_codelist, f"", 0)
        add_codeline(result_codelist, f"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)", 0)
        add_codeline(result_codelist, f"#include <sm_61_intrinsics.h>", 0)
        add_codeline(result_codelist, f"#endif", 0)
        add_codeline(result_codelist, f"#include <mma.h>", 0)


        #Support only one kernel for now
        add_codeline(result_codelist, kernel_intro[0], 0)
        #Nasty getter of variable name for now
        #TODO:Make it neat
        input_featuremap_name = kernel_intro[0].split(" ")[7][:-1]
        input_kernel_name = kernel_intro[0].split(" ")[10][:-1]
        output_featuremap_name = kernel_intro[0].split(" ")[13][:-1]

        ################################################
        ################################################
        ##### Initial declaration & initialization #####
        ################################################
        ################################################

        axis_order = ["ic_outer","kh","ic_inner","kw", "tile"]
        axis_size = {"kh": KERNEL_SIZE, "kw": KERNEL_SIZE, "ic_outer": IC_OUTER, "ic_inner": IC_INNER}

        vectorized_load = True 
        ko_kh_reorder = REORDER
        register_level_packing = True
        #kw_ki_reorder = False
        if ko_kh_reorder:
            axis_order[0], axis_order[1] = axis_order[1], axis_order[0]
        #if kw_ki_reorder:
            #axis_order[2], axis_order[3] = axis_order[3], axis_order[2]

        memory_layout = dict()
        memory_layout["featuremap_global"] = [("kh", FEATUREMAP_HEIGHT), ("kw", FEATUREMAP_WIDTH), ("ic_outer", IC_OUTER), ("ic_inner", IC_INNER), ("tile", self.wmma_m * self.wmma_k // PACK_RATE)]
        memory_layout["featuremap_shared_dummy"] = [("kh", KERNEL_SIZE), ("kw", TB_ROW_COVER + KERNEL_SIZE - 1), ("ic_outer", IC_OUTER), ("ic_inner", IC_INNER), ("tile", self.wmma_m * self.wmma_k // PACK_RATE)]

        compute_at = dict()
        compute_at["featuremap_shared"] = "kh"

        scope = dict()
        scope["featuremap_global"] = axis_order
        scope["featuremap_shared"] = axis_order[axis_order.index(compute_at["featuremap_shared"]) + 1 :]

        memory_layout["featuremap_shared"] = []
        for (dimension_name, size) in memory_layout["featuremap_shared_dummy"]:
            if dimension_name in scope["featuremap_shared"]:
                memory_layout["featuremap_shared"].append((dimension_name, size))


        def get_dimension_size(memory_layout, target_dimension_name):
            dimension_name_list = [elem[0] for elem in memory_layout]            
            match_index = dimension_name_list.index(target_dimension_name)
            return memory_layout[match_index][1]
        
        def get_dimension_base(memory_layout, target_dimension_name):
            multiplier = 1
            for (dimension_name, size) in memory_layout[::-1]:
                if(dimension_name == target_dimension_name):
                    return int(multiplier)
                multiplier *= size            
            assert(False)





        accum_fragments = (M_WARP * N_WARP) // (self.wmma_m * self.wmma_n)
        #featuremap_shared_size = (TB_ROW_COVER + KERNEL_SIZE - 1) * K_TB

        featuremap_shared_size = 1
        for (axis, multiplier) in memory_layout["featuremap_shared"]:
            if axis in scope["featuremap_shared"]:
                featuremap_shared_size *= multiplier


        featuremap_shared_size = int(featuremap_shared_size)

        featuremap_fragments = (M_WARP * K_WARP) // (self.wmma_m * self.wmma_k * IC_OUTER)
        if register_level_packing:
            kernel_shared_size = max((K_TB * N_TB) // (PACK_RATE * IC_OUTER), (M_TB * N_TB) // PACK_RATE)
        else:
            kernel_shared_size = max((K_TB * N_TB) // (PACK_RATE * IC_OUTER), (M_TB * N_TB))
        kernel_fragments = (K_WARP * N_WARP) // (self.wmma_k * self.wmma_n * IC_OUTER)

        warp_size = 32
        whole_block_load_size = (block_col_warps * block_row_warps * warp_size)
        inner_block_load_size = (block_row_warps * warp_size)
        whole_warp_load_size = warp_size

        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::accumulator, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, int> Conv_wmma_accumulator[{accum_fragments}];")
        add_codeline(result_codelist, f"__shared__ int featuremap_shared[{featuremap_shared_size}];")
        if manual_correctness_check:
            ########using u4 on featuremap for correctness check
            add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> featuremap_frag[{featuremap_fragments}];")
        else:
            add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::row_major> featuremap_frag[{featuremap_fragments}];")
        add_codeline(result_codelist, f"__shared__ int kernel_shared[{kernel_shared_size}];")
        add_codeline(result_codelist, f"nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, {self.wmma_m}, {self.wmma_n}, {self.wmma_k}, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> kernel_frag[{kernel_fragments}];")
        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int o_c_init = 0; o_c_init < {accum_fragments}; ++o_c_init) {{")
        add_codeline(result_codelist, f"(void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);", 2)
        add_codeline(result_codelist, f"}}")
        add_codeline(result_codelist, f"int outfeature_row = (blockIdx.x * {TB_ROW_COVER}) / {FEATUREMAP_WIDTH};")
        add_codeline(result_codelist, f"int outfeature_col = (blockIdx.x * {TB_ROW_COVER}) % {FEATUREMAP_WIDTH};")

        ################################################
        ################################################
        ################## First loop ##################
        ################################################
        ################################################

        first_axis_name = axis_order[0]
        first_axis_size = axis_size[first_axis_name]
        add_codeline(result_codelist, f"#pragma unroll")
        add_codeline(result_codelist, f"for (int {first_axis_name} = 0; {first_axis_name} < {first_axis_size}; {first_axis_name}++) {{")


        #block_col_warps, block_row_warps, warp_size

        if compute_at["featuremap_shared"] == first_axis_name:
            global_base_addr_string = f"int featuremap_global_base = (outfeature_row * {FEATUREMAP_WIDTH} * {IC_OUTER} * {IC_INNER} * {self.wmma_m * self.wmma_k // PACK_RATE}"
            global_base_addr_string += f" + outfeature_col * {IC_OUTER} * {IC_INNER} * {self.wmma_m * self.wmma_k // PACK_RATE}"
            first_axis_base = get_dimension_base(memory_layout["featuremap_global"], first_axis_name)
            global_base_addr_string += f" + {first_axis_name} * {first_axis_base})"
            if vectorized_load:
                global_base_addr_string += f"/4"
            global_base_addr_string += f";"
            add_codeline(result_codelist, global_base_addr_string, 2)
            

            if vectorized_load:          
                featuremap_shared_size_vectorized = featuremap_shared_size//4
                load_iter = -(featuremap_shared_size_vectorized // -whole_block_load_size)
            else:
                load_iter = -(featuremap_shared_size // -whole_block_load_size)

            add_codeline(result_codelist, "#pragma unroll", 2)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{", 2)
            add_codeline(result_codelist, f"int addressing_space = load_iter * {block_col_warps} * {block_row_warps} * {warp_size} + threadIdx.z * {block_row_warps} * {warp_size} + threadIdx.y * {warp_size} + threadIdx.x;", 3)
            
            dimension_names_shared = [dimension_name for (dimension_name, size) in memory_layout["featuremap_shared"]]
            dimension_names_shared = dimension_names_shared[::-1]
            denominator = 1
            for dimension_name in dimension_names_shared:
                dimension_size = get_dimension_size(memory_layout["featuremap_shared"], dimension_name)
                if vectorized_load and dimension_name == "tile":
                    dimension_size //= 4
                #add_codeline(result_codelist, f"int {dimension_name}_dimension = (addressing_space / {denominator}) % {dimension_size};", 3)
                if dimension_name == "kw":
                    add_codeline(result_codelist, f"int {dimension_name}_dimension = (addressing_space / {denominator});", 3)
                denominator *= dimension_size
            add_codeline(result_codelist, f"bool out_of_shmem_bound = addressing_space > {denominator};", 3)
            add_codeline(result_codelist, "if (out_of_shmem_bound)", 3)
            add_codeline(result_codelist, "break;", 4)

            dimension_name = dimension_names_shared[0]
            dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)            
            dst_addr_line = f"int dst_addr = {dimension_name}_dimension * {dimension_base}"
            for dimension_name in dimension_names_shared[1:]:
                dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
                if vectorized_load and dimension_name != "tile":
                    dimension_base //= 4
                dst_addr_line += f" + {dimension_name}_dimension * {dimension_base}"
            dst_addr_line += ";"
            #add_codeline(result_codelist, dst_addr_line, 3)


            dimension_names_global = [dimension_name for (dimension_name, size) in memory_layout["featuremap_global"]]
            dimension_names_global = dimension_names_global[::-1]            
            src_addr_line = "int src_addr = featuremap_global_base"
            for dimension_name in dimension_names_global:
                # current load scope is on featuremap_shared
                if dimension_name in scope["featuremap_shared"]:
                    dimension_base = get_dimension_base(memory_layout["featuremap_global"], dimension_name)
                    if vectorized_load and dimension_name != "tile":
                        dimension_base //= 4
                    src_addr_line += f" + {dimension_name}_dimension * {dimension_base}"
            src_addr_line += f" - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC}"
            if vectorized_load:
                src_addr_line += " / 4"
            src_addr_line += ";"
            #add_codeline(result_codelist, src_addr_line, 3)

            add_codeline(result_codelist, f"int dst_addr = addressing_space;", 3)
            if vectorized_load:
                add_codeline(result_codelist, f"int src_addr = featuremap_global_base + addressing_space - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC} / 4;", 3)
            else:
                add_codeline(result_codelist, f"int src_addr = featuremap_global_base + addressing_space - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC});", 3)


            if ko_kh_reorder:
                kh_axis_name = "kh"
            else:
                kh_axis_name = "kh_dimension"
            add_codeline(result_codelist, f"bool inside_gmem_bound = ({PADDING} <= (outfeature_row + {kh_axis_name})) && ((outfeature_row + {kh_axis_name}) < {PADDING} + {FEATUREMAP_HEIGHT});",3)
            add_codeline(result_codelist, f"inside_gmem_bound &= ({PADDING} <= (outfeature_col + kw_dimension)) && ((outfeature_col + kw_dimension) < {PADDING} + {FEATUREMAP_WIDTH});",3)
            if vectorized_load:
                add_codeline(result_codelist, f"int4 zeros = {{0,0,0,0}};",3)
                add_codeline(result_codelist, f"((int4*)featuremap_shared)[dst_addr] = (inside_gmem_bound) ? ((int4*){input_featuremap_name})[src_addr] : zeros;", 3)
            else:
                add_codeline(result_codelist, f"featuremap_shared[dst_addr] = (inside_gmem_bound) ? ((int*){input_featuremap_name})[src_addr] : 0;", 3)
            add_codeline(result_codelist, f"}}",2)
            add_codeline(result_codelist, f"__syncthreads();",2)


        ################################################
        ################################################
        ################# Second loop ##################
        ################################################
        ################################################

        second_axis_name = axis_order[1]
        second_axis_size = axis_size[second_axis_name]
        add_codeline(result_codelist, f"#pragma unroll", 2)
        add_codeline(result_codelist, f"for (int {second_axis_name} = 0; {second_axis_name} < {second_axis_size}; {second_axis_name}++) {{", 2)

        if compute_at["featuremap_shared"] == second_axis_name:
            global_base_addr_string = f"int featuremap_global_base = (outfeature_row * {FEATUREMAP_WIDTH} * {IC_OUTER} * {IC_INNER} * {self.wmma_m * self.wmma_k // PACK_RATE}"
            global_base_addr_string += f" + outfeature_col * {IC_OUTER} * {IC_INNER} * {self.wmma_m * self.wmma_k // PACK_RATE}"

            first_axis_base = get_dimension_base(memory_layout["featuremap_global"], first_axis_name)
            global_base_addr_string += f" + {first_axis_name} * {first_axis_base}"

            second_axis_base = get_dimension_base(memory_layout["featuremap_global"], second_axis_name)
            global_base_addr_string += f" + {second_axis_name} * {second_axis_base})"
            if vectorized_load:
                global_base_addr_string += "/4"
            global_base_addr_string += ";"

            add_codeline(result_codelist, global_base_addr_string, 3)


            if vectorized_load:
                featuremap_shared_size_vectorized = featuremap_shared_size//4
                load_iter = -(featuremap_shared_size_vectorized // -whole_block_load_size)
            else:
                load_iter = -(featuremap_shared_size // -whole_block_load_size)

            add_codeline(result_codelist, "#pragma unroll",3)
            add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{", 3)
            add_codeline(result_codelist, f"int addressing_space = load_iter * {block_col_warps} * {block_row_warps} * {warp_size} + threadIdx.z * {block_row_warps} * {warp_size} + threadIdx.y * {warp_size} + threadIdx.x;" , 4)

            dimension_names_shared = [dimension_name for (dimension_name, size) in memory_layout["featuremap_shared"]]
            dimension_names_shared = dimension_names_shared[::-1]
            denominator = 1
            for dimension_name in dimension_names_shared:
                dimension_size = get_dimension_size(memory_layout["featuremap_shared"], dimension_name)
                if vectorized_load and dimension_name == "tile":
                    dimension_size //= 4
                #add_codeline(result_codelist, f"int {dimension_name}_dimension = (addressing_space / {denominator}) % {dimension_size};", 4)
                if dimension_name == "kw":
                    add_codeline(result_codelist, f"int {dimension_name}_dimension = (addressing_space / {denominator});", 4)
                denominator *= dimension_size
            add_codeline(result_codelist, f"bool out_of_shmem_bound = addressing_space > {denominator};", 4)
            add_codeline(result_codelist, "if (out_of_shmem_bound)", 4)
            add_codeline(result_codelist, "break;", 5)

            dimension_name = dimension_names_shared[0]
            dimension_base  = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
            dst_addr_line = f"int dst_addr = {dimension_name}_dimension * {dimension_base}"
            for dimension_name in dimension_names_shared[1:]:
                dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
                if vectorized_load and dimension_name != "tile":
                    dimension_base //= 4
                dst_addr_line += f" + {dimension_name}_dimension * {dimension_base}"
            dst_addr_line += ";"
            #add_codeline(result_codelist, dst_addr_line, 4)


            dimension_names_global = [dimension_name for (dimension_name, size) in memory_layout["featuremap_global"]]
            dimension_names_global = dimension_names_global[::-1]
            src_addr_line = "int src_addr = featuremap_global_base"
            for dimension_name in dimension_names_global:
                #current load space is on featuremap_shared
                if dimension_name in scope["featuremap_shared"]:
                    dimension_base = get_dimension_base(memory_layout["featuremap_global"], dimension_name)
                    if vectorized_load and dimension_name != "tile":
                        dimension_base //= 4
                    src_addr_line += f" + {dimension_name}_dimension * {dimension_base}"
            src_addr_line += f" - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC}"
            if vectorized_load:
                src_addr_line += " / 4"
            src_addr_line += ";"
            #add_codeline(result_codelist, src_addr_line,4)

            add_codeline(result_codelist, f"int dst_addr = adressing_space;", 3)
            if vectorized_load:
                add_codeline(result_codelist, f"int src_addr = featuremap_global_base + addressing_space - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC} / 4;", 3)
            else:
                add_codeline(result_codelist, f"int src_addr = featuremap_global_base + addressing_space - ({PADDING} * {FEATUREMAP_WIDTH} + {PADDING}) * {NUM_IC});", 3)

            add_codeline(result_codelist, f"bool inside_gmem_bound = ({PADDING} <= (outfeature_row + kh)) && ((outfeature_row + kh) < {PADDING} * {FEATUREMAP_HEIGHT});", 4)
            add_codeline(result_codelist, f"inside_gmem_bound &= ({PADDING} <= (outfeature_col + kw_dimension)) && ((outfeature_col + kw_dimension) < {PADDING} * {FEATUREMAP_WIDTH});", 4)
            if vectorized_load:
                add_codeline(result_codelist, f"int4 zeros = {{0,0,0,0}};", 3)
                add_codeline(result_codelist, f"((int4*)featuremap_shared)[dst_addr] = (inside_gmem_bound) ? ((int4*){input_featuremap_name})[src_addr] : zeros;", 4)
            else:
                add_codeline(result_codelist, f"featuremap_shared[dst_addr] = (inside_gmem_bound) ? ((int*){input_featuremap_name})[src_addr] : 0;", 4)
            add_codeline(result_codelist, f"}}", 3)
            add_codeline(result_codelist, f"__syncthreads();", 3)


        ################################################
        ################################################
        ################# Third loop  ##################
        ################################################
        ################################################


        add_codeline(result_codelist, f"#pragma unroll",3)
        add_codeline(result_codelist, f"for (int kw = 0; kw < {KERNEL_SIZE}; kw++) {{",3)
        add_codeline(result_codelist, f"__syncthreads();",4)


        ################################################
        ################################################
        ######## Load featuremap to fragments ##########
        ################################################
        ################################################

        add_codeline(result_codelist, f"if (kw==0) {{",4)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for(int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",5)
        add_codeline(result_codelist, f"int shared_mem_idx = (threadIdx.y * {warp_row_tiles} + row_iter);",6)
        add_codeline(result_codelist, f"#pragma unroll",6)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",6)

        featuremap_load_str = f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[row_iter * {IC_INNER} + ic_inner], ((int *)featuremap_shared"
        dimension_names_shared = [dimension_name for (dimension_name, size) in memory_layout["featuremap_shared"]]
        dimension_names_shared = dimension_names_shared[::-1]
        denominator = 1
        for dimension_name in dimension_names_shared:
            if dimension_name == "tile":
                continue
            dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
            dimension_name = "shared_mem_idx" if dimension_name == "kw" else dimension_name
            featuremap_load_str += f" + {dimension_name} * {dimension_base}"
        featuremap_load_str += f"), 32);"
        add_codeline(result_codelist, featuremap_load_str, 7)
        #add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[row_iter * {IC_INNER} + ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);", 7)
        add_codeline(result_codelist, f"}}",5)
        add_codeline(result_codelist, f"}}",4)

        add_codeline(result_codelist, f"else {{",4)
        add_codeline(result_codelist, f"int shared_mem_idx = ((threadIdx.y + 1) * {warp_row_tiles} + kw - 1);",5)
        add_codeline(result_codelist, f"#pragma unroll",5)
        add_codeline(result_codelist, f"for(int ic_inner = 0; ic_inner < {IC_INNER}; ic_inner++)",5)
        if warp_row_tiles != 1:
            featuremap_load_str = f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[(kw - 1) * {IC_INNER} + ic_inner], ((int*)featuremap_shared"
            dimension_names_shared = [dimension_name for (dimension_name, size) in memory_layout["featuremap_shared"]]
            dimension_names_shared = dimension_names_shared[::-1]
            denominator = 1
            for dimension_name in dimension_names_shared:
                if dimension_name == "tile":
                    continue
                dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
                dimension_name = "shared_mem_idx" if dimension_name == "kw" else dimension_name
                featuremap_load_str += f" + {dimension_name} * {dimension_base}"
            featuremap_load_str += f"), 32);"
            add_codeline(result_codelist, featuremap_load_str, 6)
            #add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[(kw - 1) * {IC_INNER} + ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);",6)

        else:
            featuremap_load_str = f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[ic_inner], ((int*)featuremap_shared"
            dimension_names_shared = [dimension_name for (dimension_name, size) in memory_layout["featuremap_shared"]]
            dimension_names_shared = dimension_names_shared[::-1]
            denominator = 1
            for dimension_name in dimension_names_shared:
                if dimension_name == "tile":
                    continue
                dimension_base = get_dimension_base(memory_layout["featuremap_shared"], dimension_name)
                dimension_name = "shared_mem_idx" if dimension_name == "kw" else dimension_name
                featuremap_load_str += f" + {dimension_name} * {dimension_base}"
            featuremap_load_str += f"), 32);"
            add_codeline(result_codelist, featuremap_load_str, 6)
            #add_codeline(result_codelist, f"(void)nvcuda::wmma::load_matrix_sync(featuremap_frag[ic_inner], ((int*)featuremap_shared + (shared_mem_idx * {IC_OUTER} * {IC_INNER} * {fragment_size}) + (ic_outer * {IC_INNER} * {fragment_size}) + (ic_inner * {fragment_size})), 32);",6)

        add_codeline(result_codelist, f"}}",4)
        add_codeline(result_codelist, f"__syncthreads();",4)




        ################################################
        ################################################
        ######## Load weight to shared memory ##########
        ################################################
        ################################################

        #should find where ic_outer is on memory space
        #weight layout = HWOIoi, H_W_{O_blocks}_{O_warps}_{O_tiles}_{IC_OUTER}_{IC_INNER}_{wmma_n}_{wmma_k}
        #global: h * w * col_blocks * block_col_warps * warp_col_tiles * IC_OUTER * IC_INNER * wmma.n * wmma.k
        #shared: block_col_warps * warp_col_tiles * IC_INNER * wmma.n * wmma.k
        ########ic_outer is base addr
        global_kh_dimension_size = KERNEL_SIZE * NUM_OC * NUM_IC // PACK_RATE
        global_kw_dimension_size = NUM_OC * NUM_IC // PACK_RATE
        global_oc_block_size = N_TB * NUM_IC // PACK_RATE
        global_ic_outer_size = IC_INNER * self.wmma_n * self.wmma_k // PACK_RATE

        if vectorized_load:
            add_codeline(result_codelist, f"int kernel_global_base = ((kh * {global_kh_dimension_size}) + (kw * {global_kw_dimension_size}) + (blockIdx.y * {global_oc_block_size}) + (ic_outer * {global_ic_outer_size})) / 4;",4)
        else:
            add_codeline(result_codelist, f"int kernel_global_base = (kh * {global_kh_dimension_size}) + (kw * {global_kw_dimension_size}) + (blockIdx.y * {global_oc_block_size}) + (ic_outer * {global_ic_outer_size});",4)

        if vectorized_load:
            kernel_shared_size_vectorized = kernel_shared_size // 4
            load_iter = -(kernel_shared_size_vectorized // -whole_block_load_size)
        else:
            load_iter = -(kernel_shared_size // -whole_block_load_size)
        add_codeline(result_codelist, f"#pragma unroll",4)
        add_codeline(result_codelist, f"for (int load_iter = 0; load_iter < {load_iter}; load_iter++) {{",4)
        add_codeline(result_codelist, f"int addressing_space = load_iter * {block_col_warps} * {block_row_warps} * {warp_size} + threadIdx.z * {block_row_warps} * {warp_size} + threadIdx.y * {warp_size} + threadIdx.x;",5)

        denominator = 1
        tile_size = self.wmma_n * self.wmma_k // PACK_RATE
        if vectorized_load:
            tile_size //= 4
        #add_codeline(result_codelist, f"int tile_dimension = (addressing_space / {denominator}) % {tile_size};",5)

        denominator *= tile_size
        ic_inner_size = IC_INNER
        #add_codeline(result_codelist, f"int ic_inner_dimension = (addressing_space / {denominator}) % {ic_inner_size};",5)

        denominator *= ic_inner_size
        add_codeline(result_codelist, f"int ic_dimension = addressing_space % {denominator};",5) 


        output_channel_size = block_col_warps * warp_col_tiles
        #add_codeline(result_codelist, f"int output_channel_dimension = (addressing_space / {denominator}) % {output_channel_size};",5)
        add_codeline(result_codelist, f"int output_channel_dimension = (addressing_space / {denominator});",5)

        denominator *= output_channel_size
        add_codeline(result_codelist, f"bool out_of_bound = addressing_space > {denominator};", 5)
        add_codeline(result_codelist, f"if (out_of_bound)", 5)
        add_codeline(result_codelist, f"break;", 6)
        #add_codeline(result_codelist, f"int kernel_dst_addr = tile_dimension * 1 + ic_inner_dimension * {tile_size} + output_channel_dimension * {ic_inner_size} * {tile_size};", 5)
        #add_codeline(result_codelist, f"int kernel_src_addr = kernel_global_base + tile_dimension * 1 + ic_inner_dimension * {tile_size} + output_channel_dimension * {IC_OUTER} * {IC_INNER} * {tile_size};", 5)
        add_codeline(result_codelist, f"int kernel_dst_addr = ic_dimension + output_channel_dimension * {ic_inner_size} * {tile_size};", 5)
        add_codeline(result_codelist, f"int kernel_src_addr = kernel_global_base + ic_dimension + output_channel_dimension * {IC_OUTER} * {IC_INNER} * {tile_size};", 5)
        if vectorized_load:
            add_codeline(result_codelist, f"((int4*)kernel_shared)[kernel_dst_addr] = ((int4*){input_kernel_name})[kernel_src_addr];", 5)
        else:
            add_codeline(result_codelist, f"kernel_shared[kernel_dst_addr] = ((int*){input_kernel_name})[kernel_src_addr];", 5)
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



        if register_level_packing:
            ################################################
            ################################################
            ##### Conduct fragment level data packing ######
            ################################################
            ################################################
            #each warp should cover at least four fragment tiles on output channel level for warp-level register packing
            warp_register_count = 32
            packed_tile_register_count = self.wmma_m * self.wmma_n // PACK_RATE

            tiles_for_packing = warp_register_count // packed_tile_register_count

            if not (warp_col_tiles % tiles_for_packing == 0):
                return "Fallback"
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
            if manual_correctness_check:
                #this line is required for correctness check 
                add_codeline(result_codelist, f"outval = min(((max(outval, 0) << (long)4) * (long)1241513984 + (long)1073741824) >> (long)31, 15);", 5)
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
            add_codeline(result_codelist, f"((int*){output_featuremap_name})[global_output_dst] = kernel_shared[local_output_src];", 3)
            
            add_codeline(result_codelist, f"}}", 2)
            add_codeline(result_codelist, f"}}", 1)


        else: #implementaion without packing
            if not (warp_col_tiles % 4 == 0):
                return "Fallback"
            unpacked_tile_size = self.wmma_m * self.wmma_n
            add_codeline(result_codelist, f"#pragma unroll",1)
            add_codeline(result_codelist, f"for (int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{",1)
            add_codeline(result_codelist, f"#pragma unroll",2)
            add_codeline(result_codelist, f"for (int oc_tile = 0; oc_tile < {warp_col_tiles}; oc_tile++) {{",2)
            kernel_dst_addr = f"(int*)kernel_shared + (threadIdx.y * {warp_row_tiles}  * {block_col_warps} * {warp_col_tiles} * {unpacked_tile_size}) + (row_iter * {block_col_warps} * {warp_col_tiles} * {unpacked_tile_size}) + (threadIdx.z * {warp_col_tiles} * {unpacked_tile_size}) + (oc_tile * {unpacked_tile_size})"
            add_codeline(result_codelist, f"(void)nvcuda::wmma::store_matrix_sync({kernel_dst_addr} , Conv_wmma_accumulator[row_iter * {warp_col_tiles} + oc_tile], {self.wmma_n}, nvcuda::wmma::mem_row_major);", 3)
            add_codeline(result_codelist, f"}}",2)
            add_codeline(result_codelist, f"}}",1)

            add_codeline(result_codelist, f"#pragma unroll", 1)
            add_codeline(result_codelist, f"for (int row_iter = 0; row_iter < {warp_row_tiles}; row_iter++) {{", 1)
            channel_iter = warp_col_tiles // 4
            add_codeline(result_codelist, f"#pragma unroll", 2)
            add_codeline(result_codelist, f"for (int channel_iter = 0; channel_iter < {channel_iter}; channel_iter++) {{", 2)
            add_codeline(result_codelist, f"int packing_dst = 0;", 3)
            add_codeline(result_codelist, f"int local_output_src = (threadIdx.y * {warp_row_tiles} * {block_col_warps} * {channel_iter} * 32)",3)
            add_codeline(result_codelist, f"+ (row_iter * {block_col_warps} * {channel_iter} * 32) + (threadIdx.z * {channel_iter} * 32) + (channel_iter * 32) + (threadIdx.x);",5)
            add_codeline(result_codelist, f"local_output_src *= 8;",3)
            add_codeline(result_codelist, f"#pragma unroll", 3)
            add_codeline(result_codelist, f"for (int packing_iter = 0; packing_iter < 8; packing_iter++) {{", 3)
            add_codeline(result_codelist, f"int outval = kernel_shared[local_output_src + packing_iter];", 4)
            add_codeline(result_codelist, f"outval &= 0xf;", 4)
            add_codeline(result_codelist, f"outval = outval << 4*(7-packing_iter);", 4)
            add_codeline(result_codelist, f"packing_dst |= outval;", 4)
            add_codeline(result_codelist, f"}}", 3)
            add_codeline(result_codelist, f"int inner_batch_iter = threadIdx.x % 8;", 3)
            add_codeline(result_codelist, f"int inner_channel_iter = threadIdx.x / 8;", 3)
            add_codeline(result_codelist, f"int global_output_dst = (blockIdx.x * {block_row_warps} * {warp_row_tiles} * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",3)
            add_codeline(result_codelist, f"+ (threadIdx.y * {warp_row_tiles} * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",5)
            add_codeline(result_codelist, f"+ (row_iter * {grid_col_blocks} * {block_col_warps} * {channel_iter} * 32)",5)
            add_codeline(result_codelist, f"+ (blockIdx.y * {block_col_warps} * {channel_iter} * 32) + (threadIdx.z * {channel_iter} * 32) + (channel_iter * 32) + (inner_batch_iter * 4) + (inner_channel_iter);",5)
            add_codeline(result_codelist, f"((int*){output_featuremap_name})[global_output_dst] = packing_dst;", 3)
            add_codeline(result_codelist, f"}}", 2)
            add_codeline(result_codelist, f"}}", 1)


        add_codeline(result_codelist, f"}}", 0)

        add_codeline(result_codelist, kernel_intro[1], 0)
        add_codeline(result_codelist, f"}}", 0)

        result_code = "".join(result_codelist)

        return result_code

    def code_replace(self):
        in_code = self.code
        kernel_intro = []

        self.codegen_dict = self.generate_codegen_dict()
        if "Fallback" in self.codegen_dict:
            print("*********** Fallback ************")
            os.remove("/tmp/index.txt")
            self.dumpcode = "Fallback due to invalid tuning log"
            self.code += "//Code generation failure. Fallback to default TVM code generation\n"
            return self.code
        
        if not self.tune:
            print("====codegen_dict====")
            print(self.codegen_dict)
            print("")

        for line in in_code.splitlines():
            if "__launch_bounds__" in line:
                kernel_intro.append(line)

        #print(kernel_intro)

        result_code = self.codegen(kernel_intro)
        if result_code == "Fallback":
            print("*********** Fallback ************")
            os.remove("/tmp/index.txt")
            self.dumpcode = "Fallback due to invalid codegen parameters"
            self.code += "//Code generation failure. Fallback to default TVM code generation\n"
            return self.code

        #For now, just dump generated code
        self.dumpcode = result_code
        
        return result_code
        #return self.code

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
            self.compute_dict["in_height"] = featuremap_shape[0]
            self.compute_dict["in_width"] = featuremap_shape[1]
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

    def code_replace_with_TVM_config(self, code, cfg_data):
        
        self.tune = True
        self.code = code
        self.code += "//configcall\n"

        print("********** Code Replace from Config **********")
        # print("====cfg_data====")
        # print(cfg_data)
        # print("")

        # log_input = log_data["input"]
        tensor_shapes = cfg_data[1].workload

        featuremap_shape = tensor_shapes[1][1]
        self.compute_dict["in_height"] = featuremap_shape[0]
        self.compute_dict["in_width"] = featuremap_shape[1]
        self.compute_dict["batch"] = featuremap_shape[2]
        self.compute_dict["in_channel"] = featuremap_shape[3]

        conv_kernel_shape = tensor_shapes[2][1]
        self.compute_dict["kernel"] = conv_kernel_shape[0]
        self.compute_dict["num_filter"] = conv_kernel_shape[2]

        self.compute_dict["stride"] = tensor_shapes[3]
        self.compute_dict["padding"] = tensor_shapes[4]
        self.compute_dict["dilation"] = tensor_shapes[5]
        # print("====compute_dict====")
        # print(self.compute_dict)
        # print("")

        cfg_config = cfg_data[2]
        knobs = ['block_row_warps', 'block_col_warps', 'warp_row_tiles', 'warp_col_tiles', 'chunk'] 
        knob_list = [knob for knob in knobs]
        value_list = [cfg_config[knob].val for knob in knobs]
        knob_list.append('reorder_inner')
        value_list.append(cfg_config['reorder_inner'].perm)
        self.schedule_dict = dict(zip(knob_list, value_list))
        # print("====schedule_dict====")
        # print(self.schedule_dict)
        # print("")

        return self.code_replace()
