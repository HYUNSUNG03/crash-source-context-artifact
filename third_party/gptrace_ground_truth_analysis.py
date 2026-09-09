# Copyright 2025 Fraunhofer AISEC
# Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perform ground-truth analysis of deduplication results."""

import logging
from collections import defaultdict
from json import dumps as json_dumps
from pathlib import Path
from pickle import dump as pickle_dump

from numpy import array as np_array
from sklearn.decomposition import TruncatedSVD  # type: ignore


def get_undercounting(bug_cluster_dict: defaultdict[str, defaultdict[int, int]]) -> int:
    """Check for clusters that contain traces associated to different bugs."""
    unique_cluster_labels = {
        cluster_label
        for cluster_dict in bug_cluster_dict.values()
        for cluster_label in cluster_dict
    }

    num_undercounting = 0
    for cluster_label in unique_cluster_labels:
        bugs_assigned_to_cluster_label = [
            bug_label
            for bug_label, cluster_dict in bug_cluster_dict.items()
            if cluster_label in cluster_dict
        ]
        if len(bugs_assigned_to_cluster_label) > 1:
            num_undercounting += 1
            logging.debug(
                f"Undercounting present at cluster {cluster_label}: "
                f"{bugs_assigned_to_cluster_label}"
            )
    return num_undercounting


def get_overcounting(bug_cluster_dict: defaultdict[str, defaultdict[int, int]]) -> int:
    """Check for bugs that have traces in multiple clusters."""
    overcounting_list = [
        (bug_label, len(cluster_dict))
        for bug_label, cluster_dict in bug_cluster_dict.items()
        if len(cluster_dict) > 1
    ]
    if logging.root.level <= logging.DEBUG:
        for oc in overcounting_list:
            logging.debug(
                f"Overcounting bug_type {oc[0]}: present in {oc[1]} clusters."
            )

    return len(overcounting_list)


def get_lost(
    bug_cluster_dict: defaultdict[str, defaultdict[int, int]],
) -> tuple[int, set[str]]:
    """Check for bugs that do not have at least one pure cluster."""
    num_lost = 0
    lost_list: set[str] = set()
    for bug_label in bug_cluster_dict:
        cluster_dict = bug_cluster_dict[bug_label]
        other_bug_cluster_dict = {
            other_bug_label: bug_cluster_dict[other_bug_label]
            for other_bug_label in bug_cluster_dict
            if other_bug_label != bug_label
        }
        if all(
            [
                any(
                    [
                        cluster_label in other_cluster_dict
                        for other_cluster_dict in other_bug_cluster_dict.values()
                    ]
                )
                for cluster_label in cluster_dict
            ]
        ):
            num_lost += 1
            lost_list.add(bug_label)

    return num_lost, lost_list


def index_sets(
    cluster_list: list[int], bug_list: list[str]
) -> tuple[dict[int, set[int]], dict[str, set[int]]]:
    """Compute index lists that are needed for statistical computations."""
    index_cluster: dict[int, set[int]] = {}
    for cluster in set(cluster_list):
        index_cluster[cluster] = {
            n
            for n, other_cluster in enumerate(cluster_list)
            if other_cluster == cluster
        }

    index_bug: dict[str, set[int]] = {}
    for bug in set(bug_list):
        index_bug[bug] = {n for n, other_bug in enumerate(bug_list) if other_bug == bug}

    return index_cluster, index_bug


def purity(
    cluster_list: list[int],
    bug_list: list[str],
    index_cluster: dict[int, set[int]],
    index_bug: dict[str, set[int]],
    num_decimal_places: int = 5,
) -> float:
    """Compute purity."""
    purity: float = 0
    N = len(cluster_list)

    for i in set(cluster_list):
        C_i = cluster_list.count(i)
        index_cluster_i = index_cluster[i]

        maxP: float = 0
        for j in set(bug_list):
            index_bug_j = index_bug[j]

            intersection = len(index_cluster_i & index_bug_j)

            P = intersection / C_i
            if maxP < P:
                maxP = P

        purity += maxP * C_i / N

    return round(purity, num_decimal_places)


def inverse_purity(
    cluster_list: list[int],
    bug_list: list[str],
    index_cluster: dict[int, set[int]],
    index_bug: dict[str, set[int]],
    num_decimal_places: int = 5,
) -> float:
    """Compute inverse purity."""
    inverse_purity: float = 0
    N = len(cluster_list)

    for i in set(bug_list):
        L_i = bug_list.count(i)
        index_bug_i = index_bug[i]

        maxP: float = 0
        for j in set(cluster_list):
            index_cluster_j = index_cluster[j]

            intersection = len(index_bug_i & index_cluster_j)

            P = intersection / L_i
            if maxP < P:
                maxP = P

        inverse_purity += maxP * L_i / N

    return round(inverse_purity, num_decimal_places)


def f_measure(
    cluster_list: list[int],
    bug_list: list[str],
    index_cluster: dict[int, set[int]],
    index_bug: dict[str, set[int]],
    num_decimal_places: int = 5,
) -> float:
    """Compute F-measure."""
    f_measure: float = 0
    N = len(cluster_list)

    C_counts: dict[int, int] = {}
    for j in set(cluster_list):
        C_counts[j] = cluster_list.count(j)

    for i in set(bug_list):
        L_i = bug_list.count(i)
        index_bug_i = index_bug[i]

        maxF: float = 0
        for j in set(cluster_list):
            C_j = C_counts[j]
            index_cluster_j = index_cluster[j]

            intersection = len(index_bug_i & index_cluster_j)

            R = intersection / C_j
            P = intersection / L_i

            if R == 0 and P == 0:
                F: float = 0
            else:
                F = 2 * R * P / (R + P)

            if maxF < F:
                maxF = F

        f_measure += maxF * L_i / N

    return round(f_measure, num_decimal_places)


def statistical_scores(
    cluster_list: list[int], bug_list: list[str]
) -> tuple[float, float, float]:
    """Compute purity, inverse purity and F-measure."""
    index_cluster, index_bug = index_sets(cluster_list, bug_list)

    p = purity(cluster_list, bug_list, index_cluster, index_bug)
    ip = inverse_purity(cluster_list, bug_list, index_cluster, index_bug)
    f = f_measure(cluster_list, bug_list, index_cluster, index_bug)

    return (p, ip, f)


def get_dist_data(
    cluster_labels: list[int],
    unique_trace_paths: list[Path],
    duplicates: dict[Path, list[Path]],
) -> tuple[defaultdict[str, defaultdict[int, int]], list[int], list[str]]:
    """Produce various structures that contain distribution information."""
    # The number of assigned labels should be equal to the number of unique stack traces
    assert len(cluster_labels) == len(unique_trace_paths)

    # Dictionary that, for each bug type, stores how many associated traces have been
    # assigned to each individual cluster.
    bug_cluster_dict: defaultdict[str, defaultdict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # List that contain for all (also duplicate) traces the clustering label and the
    # ground-truth label, in the same order
    cluster_list: list[int] = []
    bug_list: list[str] = []

    for i, trace_path in enumerate(unique_trace_paths):
        bug_label = trace_path.parent.name
        cluster_label = cluster_labels[i]
        bug_cluster_dict[bug_label][cluster_label] += 1

        cluster_list.append(cluster_label)
        bug_list.append(bug_label)

        for duplicate_path in duplicates[trace_path]:
            bug_label = duplicate_path.parent.name
            bug_cluster_dict[bug_label][cluster_label] += 1

            cluster_list.append(cluster_label)
            bug_list.append(bug_label)

    return bug_cluster_dict, cluster_list, bug_list


def decimal_to_int_percentage(x: float) -> int:
    """Convert decimal value to integer percentage."""
    return int(100 * x + 0.5)


def analyze_clustering_performance(
    cluster_labels: list[int],
    unique_trace_paths: list[Path],
    duplicates: dict[Path, list[Path]],
    num_clusters: int,
    summary_path: Path,
    round: bool,
):
    """Do ground-truth analysis."""
    bug_cluster_dict, cluster_list, bug_list = get_dist_data(
        cluster_labels, unique_trace_paths, duplicates
    )

    num_undercounting = get_undercounting(bug_cluster_dict)
    num_overcounting = get_overcounting(bug_cluster_dict)
    num_lost, lost_list = get_lost(bug_cluster_dict)
    p, ip, f = statistical_scores(cluster_list, bug_list)

    if round:
        p = decimal_to_int_percentage(p)
        ip = decimal_to_int_percentage(ip)
        f = decimal_to_int_percentage(f)

    ground_truth_results = {
        "num_clusters": num_clusters,
        "num_overcount": num_overcounting,
        "num_undercount": num_undercounting,
        "num_completely_lost": num_lost,
        "purity": p,
        "inverse_purity": ip,
        "f_measure": f,
    }

    with open(summary_path, "w") as summary:
        logging.info(f"Ground truth analysis:\n{json_dumps(ground_truth_results)}")
        summary.write(json_dumps(ground_truth_results) + "\n")
        for b in lost_list:
            logging.info(f"Bug {b} has no distinct cluster and will be lost")
            summary.write(f"Bug {b} has no distinct cluster and will be lost\n")


def store_projections(
    projections_path: Path,
    projections_dict: dict[str, list[list[float]]],
    indices: list[Path],
):
    """Store two-dimensional projections of embeddings."""
    final = {}
    for key in projections_dict:
        tsne = TruncatedSVD(n_components=2)
        vis_dims = tsne.fit_transform(np_array(projections_dict[key]))

        for i, reduced in enumerate(vis_dims):
            sci = indices[i]
            # Not that the bug key only makes sense if the traces are stored
            # according to ground-truth labels
            if sci not in final:
                final[sci] = {"bug": str(sci.parent)}

            final[sci][key] = reduced

    with open(projections_path, "wb") as f:
        pickle_dump(final, f)
