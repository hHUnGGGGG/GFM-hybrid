# Path Interpretations

GFM-RAG exhibits the multi-hop reasoning ability powered by the multi-layer GFM. You can visualize the paths captured by the GFM-RAG to provide path interpretations for the prediction.

The paths' importance to the final prediction can be quantified by the partial derivative of the prediction score with respect to the triples at each layer (hop), defined as:

\[
    s_1,s_2,\ldots,s_L=\arg\mathop{\text{top-}}k\frac{\partial p_e(q)}{\partial s_l}
\]

The top-$k$ path interpretations can be obtained by the top-$k$ longest paths with beam search.

You can visualize the paths and their importance using the following code:

??? example "gfmrag/workflow/config/exp_visualize_path.yaml"

    ```yaml title="gfmrag/workflow/config/exp_visualize_path.yaml"
    --8<-- "gfmrag/workflow/config/exp_visualize_path.yaml"
    ```

??? example "gfmrag/workflow/experiments/visualize_path.py"

    <!-- blacken-docs:off -->
    ```python title="gfmrag/workflow/experiments/visualize_path.py"
    --8<--"gfmrag/workflow/experiments/visualize_path.py"
    ```
    <!-- blacken-docs:on -->

```bash
python -m gfmrag.workflow.experiments.visualize_path
```

Path Examples

```
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Question: What football club was owned by the singer of "Grow Some Funk of Your Own"?
Answer: Watford Football Club
Question Entities: ['grow some funk of your own', 'football club']
Supporting Facts: ['Grow Some Funk of Your Own', 'Elton John']
Predicted Paths:
--------------------------------------------------------
Target Entity: watford football club
1.0950: [ grow some funk of your own, is a song by, elton john ] => [ elton john, equivalent, sir elton hercules john ] => [ sir elton hercules john, inverse_named a stand after, watford football club ]
0.9154: [ grow some funk of your own, is a song by, elton john ] => [ elton john, equivalent, sir elton hercules john ] => [ sir elton hercules john, owned, watford football club ]
```
