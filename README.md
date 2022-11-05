# TSP-GeneAl 使用概述
A. 本程式利用Python 軟體套件 geneal 執行旅行業務員問題(Travelling salesman problem, TSP)。

  1. 本程式可選擇旅行國家及/或城市透過geneal所提供之基因演算法及其參數之設定，執行最佳路徑之搜索。
  2. 參數之設定影響演算法執行之效能，包括其最佳性、執行時間及使用記憶空間。
  3. 參數包括 (1) 選秀交配策略(selection_strategy) (2) 獲選機率(selection_rate) (3) 突變策略(mutation_strategy) (4) 突變機率   (mutation_rate) (5) 族群規模(pop_size) (6) 遺傳代數(max_gen)

B. 執行本程式需先安裝的PYPI套件:
  1. numpy
  2. geneal
  3. turfpy
  4. pandas
  5. plotly
  6. numba
  7. matplotlib

C. Reference fot GA for TSP Problem: 
   http://www.inf.tu-dresden.de/content/institutes/ki/cl/study/summer14/pssai/slides/GA_for_TSP.pdf

