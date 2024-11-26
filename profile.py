import itertools
import pandas as pd
from typing import List, Dict, Callable, Any
import abc
import triton


class BenchmarkCandidate:
    @abc.abstractmethod
    def benchmark_content(self):
        pass

BenchmarkCandidateFactory = Callable[[Any], BenchmarkCandidate]


def benchmark(candidates: Dict[str, BenchmarkCandidateFactory], bench_input_list: List[Dict[str, Any]]) -> pd.DataFrame:
    results = []
    for bench_input_dict in bench_input_list:
        for name, candidate_factory in candidates.items():
            candidate_ready: BenchmarkCandidate = candidate_factory(**bench_input_dict)
            ms, min_ms, max_ms = triton.testing.do_bench(
                candidate_ready.benchmark_content, quantiles=[0.5, 0.2, 0.8]
            )

            results.append({
                "name": name,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                **bench_input_dict
            })
    
    return pd.DataFrame(results)
    

def params_grid_to_list(params_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = params_grid.keys()
    values = params_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def pivot(df: pd.DataFrame, index: List[str], columns: str, values: str) -> pd.DataFrame:
    return df.pivot(index=index, columns=columns, values=values)

