
## Simple Ranker

!!! example

    ```yaml title="gfmrag_hybrid/workflow/config/doc_ranker/simple_ranker.yaml"
    --8<-- "gfmrag_hybrid/workflow/config/doc_ranker/simple_ranker.yaml"
    ```

| Parameter  |              Options              |                               Note                                |
| :--------: | :-------------------------------: | :---------------------------------------------------------------: |
| `_target_` | `gfmrag_hybrid.doc_rankers.SimpleRanker` | The class name of [SimpleRanker][gfmrag_hybrid.doc_rankers.SimpleRanker] |

## IDF Ranker

!!! example

    ```yaml title="gfmrag_hybrid/workflow/config/doc_ranker/idf_ranker.yaml"
    --8<-- "gfmrag_hybrid/workflow/config/doc_ranker/idf_ranker.yaml"
    ```

| Parameter  |                Options                 |                                     Note                                     |
| :--------: | :------------------------------------: | :--------------------------------------------------------------------------: |
| `_target_` | `gfmrag_hybrid.doc_rankers.IDFWeightedRanker` | The class name of [IDFWeightedRanker ][gfmrag_hybrid.doc_rankers.IDFWeightedRanker] |

## Top-k Ranker

!!! example

    ```yaml title="gfmrag_hybrid/workflow/config/doc_ranker/topk_ranker.yaml"
    --8<-- "gfmrag_hybrid/workflow/config/doc_ranker/topk_ranker.yaml"
    ```

| Parameter  |             Options             |                             Note                              |
| :--------: | :-----------------------------: | :-----------------------------------------------------------: |
| `_target_` | `gfmrag_hybrid.doc_rankers.TopKRanker` | The class name of [TopKRanker][gfmrag_hybrid.doc_rankers.TopKRanker] |
|  `top_k`   |             Integer             |        The top-k entities used for document retrieval         |

## IDF Top-k Ranker

!!! example

    ```yaml title="gfmrag_hybrid/workflow/config/doc_ranker/idf_topk_ranker.yaml"
    --8<-- "gfmrag_hybrid/workflow/config/doc_ranker/idf_topk_ranker.yaml"
    ```

| Parameter  |                  Options                   |                                        Note                                         |
| :--------: | :----------------------------------------: | :---------------------------------------------------------------------------------: |
| `_target_` | `gfmrag_hybrid.doc_rankers.IDFWeightedTopKRanker` | The class name of [IDFWeightedTopKRanker][gfmrag_hybrid.doc_rankers.IDFWeightedTopKRanker] |
|  `top_k`   |                  Integer                   |                   The top-k entities used for document retrieval                    |
