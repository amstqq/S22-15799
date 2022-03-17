# Index Recommendation

This project adapts from the github repo of paper [_Magic mirror in my hand, which is the best in the land? An Experimental Evaluation of Index Selection Algorithms_](https://github.com/hyrise/index_selection_evaluation).

## Modifications:

- Extract a subset of queries that represent the entire workload. The provided workloads for this project are comprised of hundreds of thousands of queries and it is too inefficient to process them all.
- Sample representative parameters for each query template. The sampled params need to follow the original param distribution.
- Add additional functionalities to the database connector and cost evaluator to fit the constraints of this project.
- Fine-tune configurations that optimize for the provided workloads.
- Perform visualizations on provided workloads to determine the best algorithm.

## Visualizations and Findings

### Epinions

![epinions](./graphs/epinions_goodput_vs_maxindex.png)

- The best index size appears to be around 8.
- More storage budget limit does not usually lead to better index
- Greater-sized indexes are more complex and can potentially lead to degration in performance.
