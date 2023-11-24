# Comparative Opinion Mining from Vietnamese Product Reviews

This repo contains the data sets and source code of our NLP final project.
Slide: (https://docs.google.com/presentation/d/1ZYOEaSgauSUvIJM182SJvfRREXZRvO3Czr9JSP4E5GI/edit#slide=id.g261b6a04064_0_60)

## Team members
- 21020467 Nguyễn Thị Thúy Hường
- 21020522 Hoàng Hùng Mạnh
  
## Task
This task aims to create models that can find opinions from product reviews. Each review has sentences that compare different parts of products.
<p align="center">
<img src="image/table1.png" width="50%" />
</p>

## Dataset
The dataset is released by VLSP 2023 challenge on
Comparative Opinion Mining from Vietnamese Product
Reviews. Each review contains comparative sentences,
and the corresponding quintuples are annotated.
The following table shows the statistics of the comparative quintuple corpora.
<p align="center">
<img src="image/table2.png" width="50%" />
</p>

## Approach
### Stage 1: CEE + CSI
<p align="center">
<img src="image/stage1.png" width="50%" />
</p>

### Stage 2, 3: Combination, Filtering + CLC
**Combination**
<p align="center">
<img src="image/stage2_combi.png" width="50%" />
</p>

**Filtering**
<p align="center">
<img src="image/stage2_filter.png" width="50%" />
</p>

**CLC**
<p align="center">
<img src="image/stage2_clc.png" width="50%" />
</p>

**Output**
<p align="center">
<img src="image/stage2_output.png" width="50%" />
</p>

## Result
The Results of different approaches for CEE, T4 and T5 under the Exact Match metric:
<p align="center">
<img src="image/result.png" width="50%" />
</p>

