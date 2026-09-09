# Crash source context: code and data for reproducing the results

This repository accompanies the Korean manuscript **크래시 중복 제거를 위한 주변 소스 문맥의 효과: 개발 코호트의 거리 순위와 군집 품질 평가** (draft v6). It provides saved embeddings, experiment records, and scripts for reproducing the reported analyses.

The study compares four representations of 152 crash reports from three targets: B, B+L, B+D, and B+L+D.

| Component | Description |
| --- | --- |
| B | The baseline representation, produced with a local, 384-dimensional BGE encoder. |
| L | Surrounding source code from up to three source-mapped stack frames, with a window of ±5 physical lines around each mapped location. |
| D | Static dependency candidates and extraction-status text produced by the corrected D4 implementation. |

The original experiments included source extraction, embedding, clustering, and scoring. Their frozen outputs were checked using hashes and numerical consistency checks. The study uses a development cohort; the final configuration has not been evaluated on an independent held-out dataset. B is the baseline for comparisons within this local setup, rather than a reproduction of GPTrace's published embedding configuration.

## Reproduce the results

The saved-embedding workflow has been tested with Python 3.12 and NumPy 2.3.5. It requires no account, API key, model download, compiler, or execution of crash inputs.

Run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce.py --output recomputed
```

The script checks the packaged files against the SHA-256 manifest, reconstructs 48 distance matrices from saved component embeddings, and checks the matrix hashes. It recomputes average precision (AP) and clustering F-scores, replays 96 selector decisions, and checks the scores and statistics of 960 saved grid partitions. It also generates the paper tables and the FreeType status diagnostic. Some tables summarize recorded experiments rather than rerunning them; the [table mapping](docs/TABLES.md) identifies the source of each table.

Generated files are written to `recomputed/`; the frozen experiment records are preserved. See [reproducibility scope](docs/REPRODUCIBILITY.md) for the checks performed and the inputs needed to rerun source extraction and model inference.

## Main observations

| Representation | Macro AP |
| --- | ---: |
| B | 0.9587 |
| B+L | 0.9817 |
| B+D | 0.9531 |
| B+L+D | 0.9853 |

- Across three targets and two selectors, B+L has a higher clustering F-score than B in one comparison, the same score in two, and a lower score in three.
- Across the 20 grid settings with epsilon fixed at zero, adding D to B+L increases the macro F-score in 12 settings, leaves it unchanged in four, and decreases it in four.
- In FreeType, 320 of the 330 negative pairs involved in B's ranking violations have different endpoint statuses. All 17 affected positive pairs have matching statuses.

These observations describe the development cohort. Higher AP does not establish a consistent improvement in clustering F-score. The FreeType diagnostic examines the relationship between status categories and ranking errors; it does not replace an ablation that embeds status-only text with the same encoder and combines it with B.

## Repository layout

- `data/`: frozen component embeddings, hashes, report identifiers and labels, candidate configurations, selections, and scores.
- `scripts/reproduce.py`: the tested entry point for reproducing analyses from saved embeddings.
- `src/`: frozen selector and numerical functions with workstation-specific runtime paths removed.
- `historical/`: earlier implementations and original experiment scripts for B, L, D4, and input construction. These scripts require additional inputs and environment configuration; see the [reproducibility notes](docs/REPRODUCIBILITY.md) before running them.
- `docs/`: environment details, provenance, data sources, table mapping, and experiments that remain incomplete.
- `third_party/`: the pinned GPTrace evaluation code and its Apache-2.0 license.

The repository includes the saved embeddings needed for the workflow above. It does not include raw crash inputs, the analyzed projects' source trees, dependency graphs, frozen text inputs, or model weights. Rerunning source extraction and embedding therefore requires those external inputs and the original environment configuration.

## License

The authors' original research code and derived experimental results are all rights reserved; see `LICENSE`. The bundled third-party GPTrace evaluation code remains Apache-2.0 under `third_party/GPTrace-LICENSE`.

## 한국어 안내

이 저장소는 논문 v6의 실험 코드와 결과 재현 자료를 제공합니다. 세 타깃의 크래시 보고서 152건을 대상으로 기준 표현 B에 주변 소스 문맥 L과 정적 의존성 후보 및 추출 상태 정보 D를 추가했을 때의 변화를 비교합니다.

원 실험에서는 소스 추출, 임베딩 생성, 군집화, 평가를 수행했으며, 동결한 결과의 해시와 수치 일치 여부를 확인했습니다. 공개 자료에서는 저장된 임베딩을 이용해 거리 행렬과 평가 지표를 다시 계산하고, 선택 결과와 논문 표를 확인할 수 있습니다. 위의 설치 및 실행 명령을 저장소 최상위 폴더에서 실행하면 결과가 `recomputed/`에 생성됩니다. 표별 계산 방법과 기존 실험 기록을 사용하는 범위는 [표별 재현 안내](docs/TABLES.md)에 정리되어 있습니다.

소스 추출부터 임베딩 생성까지 다시 실행하려면 분석 대상 프로젝트의 소스 코드, 의존성 그래프, 동결한 텍스트 입력, 모델 가중치 등 별도 자료와 환경 설정이 필요합니다. 이 자료들은 공개 저장소에 포함되어 있지 않습니다. 자세한 준비 사항은 [재현 범위 안내](docs/REPRODUCIBILITY.md)를 참고하세요.

주변 소스 문맥 L을 추가하면 이 개발 코호트의 평균 AP는 높아졌지만, 군집 F 점수의 일관된 개선은 확인되지 않았습니다. 최종 설정에 대한 독립적인 미관측 데이터 평가는 아직 수행하지 않았습니다. 이 연구의 B는 동일한 로컬 환경에서 비교하기 위한 기준선이며, GPTrace의 원 임베딩 설정을 재현한 결과는 아닙니다.
