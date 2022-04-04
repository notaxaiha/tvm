# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""XGBoost as cost model"""

import logging
import time
import math

import numpy as np

from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind

from .. import feature
from ..utils import get_rank
from .metric import max_curve, recall_curve, cover_curve
from .model_based_tuner import CostModel, FeatureCache

xgb = None

logger = logging.getLogger("autotvm")


class XGBoostCostModel(CostModel):
    """XGBoost as cost model

    Parameters
    ----------
    task: Task
        The tuning task
    feature_type: str, optional
        If is 'itervar', use features extracted from IterVar (loop variable).
        If is 'knob', use flatten ConfigEntity directly.
        If is 'curve', use sampled curve feature (relation feature).

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' are good.
                                'itervar' is more accurate but 'knob' is much faster.
                                There are some constraints on 'itervar', if you meet
                                problems with feature extraction when using 'itervar',
                                you can switch to 'knob'.

        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability,
                               'knob' is faster.
        For cross-device or cross-operator tuning, you can use 'curve' only.
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    upper_model: XGBoostCostModel, optional
        The upper model used in transfer learning
    """

    def __init__(
        self, task, feature_type, loss_type, num_threads=None, log_interval=25, upper_model=None
    ):
        global xgb
        super(XGBoostCostModel, self).__init__()
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostCostModel. "
                "Please install its python package first. "
                "Help: (https://xgboost.readthedocs.io/en/latest/) "
            )

        self.task = task
        self.target = task.target
        self.space = task.config_space

        self.fea_type = feature_type
        self.loss_type = loss_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads
        self.bst = None

        if feature_type == "itervar":
            self.feature_extract_func = _extract_itervar_feature_index
        elif feature_type == "cuda_ast":
            self.feature_extract_func = _extract_cuda_ast_feature_index
        elif feature_type == "knob":
            self.feature_extract_func = _extract_knob_feature_index
        elif feature_type == "curve":
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + feature_type)

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.pool = None
        self.base_model = None

        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        self.pool = PopenPoolExecutor(
            max_workers=self.num_threads,
            initializer=_extract_popen_initializer,
            initargs=(space, target, task),
        )

    def _close_pool(self):
        if self.pool:
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def _base_model_discount(self):
        return 1.0 / (2 ** (self._sample_size / 64.0))

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)

        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        if self.base_model:
            discount = self._base_model_discount()
            if discount < 0.05:  # discard base model
                self.base_model.upper_model = None
                self.base_model = None
            else:
                dtrain.set_base_margin(discount * self.base_model.predict(xs, output_margin=True))

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=8000,
            callbacks=[
                custom_callback(
                    stopping_rounds=20,
                    metric="tr-a-recall@%d" % plan_size,
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[
                        xgb_average_recalln_curve_score(plan_size),
                    ],
                    verbose_eval=self.log_interval,
                )
            ],
        )

        logger.debug(
            "XGB train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
            time.time() - tic,
            len(xs),
            len(xs) - np.sum(valid_index),
            self.feature_cache.size(self.fea_type),
        )

    def fit_log(self, records, plan_size, min_seed_records=500):
        tic = time.time()

        # filter data, only pick the data with a same task
        data = []
        for inp, res in records:
            if inp.task.name == self.task.name:
                data.append((inp, res))

        logger.debug("XGB load %d entries from history log file", len(data))

        # extract feature
        self._reset_pool(self.space, self.target, self.task)
        pool = self._get_pool()
        if self.fea_type == "itervar":
            feature_extract_func = _extract_itervar_feature_log
        elif self.fea_type == "knob":
            feature_extract_func = _extract_knob_feature_log
        elif self.fea_type == "curve":
            feature_extract_func = _extract_curve_feature_log
        else:
            raise RuntimeError("Invalid feature type: " + self.fea_type)
        result = pool.map_with_error_catching(feature_extract_func, data)

        # filter out feature with different shapes
        fea_len = len(self._get_feature([0])[0])

        xs, ys = [], []
        for res in result:
            if res.status != StatusKind.COMPLETE:
                continue
            x, y = res.value
            if len(x) == fea_len:
                xs.append(x)
                ys.append(y)

        if len(xs) < min_seed_records:  # no enough samples
            return False

        xs, ys = np.array(xs), np.array(ys)
        x_train = xs
        y_train = ys
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        plan_size *= 2
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=400,
            callbacks=[
                custom_callback(
                    stopping_rounds=100,
                    metric="tr-a-recall@%d" % plan_size,
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[
                        xgb_average_recalln_curve_score(plan_size),
                    ],
                    verbose_eval=self.log_interval,
                )
            ],
        )

        logger.debug("XGB train: %.2f\tobs: %d", time.time() - tic, len(xs))

        return True

    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)
        dtest = xgb.DMatrix(feas)

        if self.base_model:
            dtest.set_base_margin(
                self._base_model_discount() * self.base_model.predict(xs, output_margin=True)
            )

        return self.bst.predict(dtest, output_margin=output_margin)

    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return XGBoostCostModel(
            self.task, self.fea_type, self.loss_type, self.num_threads, self.log_interval, self
        )

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map_with_error_catching(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea.value if fea.status == StatusKind.COMPLETE else None

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret

    def __del__(self):
        self._close_pool()


# Global variables for passing arguments to extract functions.
_extract_space = None
_extract_target = None
_extract_task = None


def _extract_popen_initializer(space, target, task):
    global _extract_space, _extract_target, _extract_task
    _extract_space = space
    _extract_target = target
    _extract_task = task


def _extract_itervar_feature_index(args):
    """extract iteration var feature for an index in extract_space"""
    try:
        config = _extract_space.get(args)
        with _extract_target:
            sch, fargs = _extract_task.instantiate(config)

        fea = feature.get_itervar_feature_flatten(sch, fargs, take_log=True)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return fea
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_cuda_ast_feature_index(args):
    """extract iteration var feature from cuda code for an index in extract_space"""
    try:
        config = _extract_space.get(args)
        fea = get_cuda_ast_feature(config, _extract_task)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return fea
    except Exception:  # pylint: disable=broad-except
        return None

def get_cuda_ast_feature(config, task):

    knob = config.get_flatten_feature()
    # 0: block_row_warps, 1: block_col_warps, 2: warp_row_tiles, 3: warp_col_tiles, 4: chunk, 5: split_block_k_nums, 
    # 6: vector_ws, 7: vetor_as, 8: reorder, 9: AS_db, 10: WS_db, 11: auto_unroll_max_step

    dim_out = task.workload[1][1]
    dim_ker = task.workload[2][1]

    h = dim_out[0]
    w = dim_out[1]
    n = dim_out[2]
    ic = dim_out[3]

    k = dim_ker[0]
    oc = dim_ker[2]
            
    num_loops = 19
    ext = np.empty(num_loops, dtype = np.float32)
    ext[0] = knob[2] * knob[3]  # o_c_init, accum_fragments = (M_WARP * N_WARP) // (self.wmma_m * self.wmma_n) = warp_row_tiles * warp_col_tiles
    # reorder
    if knob[8] == 1:
        ext[1] = ic // 32 // knob[4]  # ic_outer, (in_channel // self.wmma_k) // chunk
        ext[2] = k  # kh, kernel size
    elif knob[9] == 0:
        ext[2] = ic // 32 // knob[4]  # ic_outer, (in_channel // self.wmma_k) // chunk
        ext[1] = k  # kh, kernel size
    
    ext[3] = 1 # load_iter for feature map, 
    
    ext[4] = k  # kw, kernel size
    ext[5] = knob[2]  # row_iter, warp_row_tiles
    ext[6] = knob[4]  # ic_inner, chunk
    
    ext[7] = -(max((ic*knob[1]*knob[3]*8)//(8*(ic//32//knob[4])),(knob[0]*knob[2]*8)*(knob[1]*knob[3]*8)//8) // 4 // -(knob[0]*knob[1]*32)) # load_iter for kernel, 
    
    ext[8] = knob[3]  # oc_tile, warp_col_tiles
    ext[9] = knob[4]  # ic_inner, chunk
    ext[10] = knob[4]  # ic_inner, chunk
    ext[11] = knob[2]  # row_iter, warp_row_tiles
    ext[12] = knob[3]  # oc_tile, warp_col_tiles
    ext[13] = knob[2]  # row_iter, warp_row_tiles
    ext[14] = knob[3] // 4  # packing_iter = warp_col_tiles // tiles_for_packing
    ext[15] =  4 # output_tile_iter, tiles_for_packing = 32 // (8 * 8 // 8) = 4
    
    ext[16] = 2 # elem_iter, Conv_wmma_accumulator[0].num_elements, 8 * 8 // 32
    
    ext[17] = knob[2]  # row_iter, warp_row_tiles
    ext[18] = knob[3] // 4  # channel_iter, warp_col_tiles // 4
    
    # _attr_ : length, nest_level, topdown, bottomup, one_hot_annotation
    # _arith_ : add_ct, mul_ct, div_ct
    # buffer : stride, mod, count, reuse, thread_count, thread_reuse
    
    fea = np.array([])
    # fea_len = 5 # length, nest_level, topdown, bottomup, stride
    
    # 0, o_c_init
    # fea = np.append(fea, np.array([])
    fea = np.append(fea, np.array([math.log2(ext[0]+1)]))
    fea = np.append(fea, np.array([1]))
    fea = np.append(fea, np.array([math.log2(ext[0]+1)]))
    fea = np.append(fea, math.log2(ext[0]+1))
    # fea = np.append(fea, np.array([1+1])

    # 1, ic_outer
    fea = np.append(fea, np.array([math.log2(ext[1]+1)]))
    fea = np.append(fea, np.array([1]))
    fea = np.append(fea, np.array([math.log2(ext[1]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]*ext[11]*ext[10]*ext[4]*ext[2]*ext[1]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 2, kh
    fea = np.append(fea, np.array([math.log2(ext[2]+1)]))
    fea = np.append(fea, np.array([2]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]*ext[11]*ext[10]*ext[4]*ext[2]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 3, load_iter
    fea = np.append(fea, np.array([math.log2(ext[3]+1)]))
    fea = np.append(fea, np.array([3]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[3]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 4, kw
    fea = np.append(fea, np.array([math.log2(ext[4]+1)]))
    fea = np.append(fea, np.array([3]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]*ext[11]*ext[10]*ext[4]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 5, row_iter
    fea = np.append(fea, np.array([math.log2(ext[5]+1)]))
    fea = np.append(fea, np.array([4]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[5]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 6, ic_inner
    fea = np.append(fea, np.array([math.log2(ext[6]+1)]))
    fea = np.append(fea, np.array([5]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[5]*ext[6]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 7, load_iter
    fea = np.append(fea, np.array([math.log2(ext[7]+1)]))
    fea = np.append(fea, np.array([4]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[7]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 8, oc_tile
    fea = np.append(fea, np.array([math.log2(ext[8]+1)]))
    fea = np.append(fea, np.array([4]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[8]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 9, ic_inner
    fea = np.append(fea, np.array([math.log2(ext[9]+1)]))
    fea = np.append(fea, np.array([5]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[8]*ext[9]+1)]))
    fea = np.append(fea, np.array([math.log2(0.5)]))
    # fea = np.append(fea, np.array([math.log2(+1)])
    
    # 10, ic_inner
    fea = np.append(fea, np.array([math.log2(ext[10]+1)]))
    fea = np.append(fea, np.array([4]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[10]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]*ext[11]*ext[10]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 11, row_iter
    fea = np.append(fea, np.array([math.log2(ext[11]+1)]))
    fea = np.append(fea, np.array([5]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[10]*ext[11]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]*ext[11]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 12, oc_tile
    fea = np.append(fea, np.array([math.log2(ext[12]+1)]))
    fea = np.append(fea, np.array([6]))
    fea = np.append(fea, np.array([math.log2(ext[1]*ext[2]*ext[4]*ext[10]*ext[11]*ext[12]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[12]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 13, row_iter
    fea = np.append(fea, np.array([math.log2(ext[13]+1)]))
    fea = np.append(fea, np.array([1]))
    fea = np.append(fea, np.array([math.log2(ext[13]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[16]*ext[15]*ext[14]*ext[13]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 14, packing_iter
    fea = np.append(fea, np.array([math.log2(ext[14]+1)]))
    fea = np.append(fea, np.array([2]))
    fea = np.append(fea, np.array([math.log2(ext[13]*ext[14]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[16]*ext[15]*ext[14]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 15, output_tile_iter
    fea = np.append(fea, np.array([math.log2(ext[15]+1)]))
    fea = np.append(fea, np.array([3]))
    fea = np.append(fea, np.array([math.log2(ext[13]*ext[14]*ext[15]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[16]*ext[15]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 16, elem_iter
    fea = np.append(fea, np.array([math.log2(ext[16]+1)]))
    fea = np.append(fea, np.array([4]))
    fea = np.append(fea, np.array([math.log2(ext[13]*ext[14]*ext[15]*ext[16]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[16]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 17, row_iter
    fea = np.append(fea, np.array([math.log2(ext[17]+1)]))
    fea = np.append(fea, np.array([1]))
    fea = np.append(fea, np.array([math.log2(ext[17]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[18]*ext[17]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    # 18, channel_iter
    fea = np.append(fea, np.array([math.log2(ext[18]+1)]))
    fea = np.append(fea, np.array([2]))
    fea = np.append(fea, np.array([math.log2(ext[17]*ext[18]+1)]))
    fea = np.append(fea, np.array([math.log2(ext[18]+1)]))
    # fea = np.append(fea, np.array([math.log2(+1)])

    fea = fea.astype(np.float32)
    
    return fea


def _extract_knob_feature_index(args):
    """extract knob feature for an index in extract_space"""
    try:

        config = _extract_space.get(args)

        return config.get_flatten_feature()
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        x = config.get_flatten_feature()

        if res.error_no == 0:
            with inp.target:  # necessary, for calculating flops of this task
                inp.task.instantiate(config)
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_curve_feature_index(args):
    """extract sampled curve feature for an index in extract_space"""
    try:

        config = _extract_space.get(args)
        with _extract_target:
            sch, fargs = _extract_task.instantiate(config)

        fea = feature.get_buffer_curve_sample_flatten(sch, fargs, sample_n=20)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return np.array(fea)
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None


def custom_callback(
    stopping_rounds, metric, fevals, evals=(), log_file=None, maximize=False, verbose_eval=True
):
    """callback function for xgboost to support multiple custom evaluation functions"""
    # pylint: disable=import-outside-toplevel
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric

    try:
        from xgboost.training import aggcv
    except ImportError:
        from xgboost.callback import _aggcv as aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state["maximize_score"] = maximize
        state["best_iteration"] = 0
        if maximize:
            state["best_score"] = float("-inf")
        else:
            state["best_score"] = float("inf")

        if bst is not None:
            if bst.attr("best_score") is not None:
                state["best_score"] = float(bst.attr("best_score"))
                state["best_iteration"] = int(bst.attr("best_iteration"))
                state["best_msg"] = bst.attr("best_msg")
            else:
                bst.set_attr(best_iteration=str(state["best_iteration"]))
                bst.set_attr(best_score=str(state["best_score"]))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(":") for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if "null" in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            logger.debug("\t".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + "\n")

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state["best_score"]
        best_iteration = state["best_iteration"]
        maximize_score = state["maximize_score"]
        if (maximize_score and score > best_score) or (not maximize_score and score < best_score):
            msg = "[%d] %s" % (env.iteration, "\t".join([_fmt_metric(x) for x in eval_res]))
            state["best_msg"] = msg
            state["best_score"] = score
            state["best_iteration"] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(
                    best_score=str(state["best_score"]),
                    best_iteration=str(state["best_iteration"]),
                    best_msg=state["best_msg"],
                )
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state["best_msg"]
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)

    return feval


def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]

    return feval


def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N

    return feval


def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]

    return feval


def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]

    return feval


def xgb_null_score(_):
    """empty score function for xgb"""

    def feval(__, ___):
        return "null", 0

    return feval
