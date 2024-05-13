# Task6.3.2
Task 6.3.2 repository for REliable &amp; eXplAinable Swarm Intelligence for People with Reduced mObility project (REXASI-PRO) (GRANT AGREEMENT NO.101070028)

We use Python 3.8.19. Necessary dependencies are in requeriments.txt


Persistence diagrams are a common tool in computational topology, especially when working with multidimensional datasets like point clouds. Dimension 0 persistence diagrams on a 2D point cloud measure the connectivity of points as the radius of a set of balls around the points increases.

Persistence diagram of dimension 0:

Connectivity: It begins by treating each point in the cloud as a separate set, then starts increasing the radius of the ball (or epsilon) around each point. As points come closer, they connect forming connected components.

Evolution: The diagram tracks how sets of points merge into larger connected components as the radius increases. Each time two sets merge, a birth-death interval is recorded for that connected component. The dimension 0 persistence diagram shows how these intervals vary with radius.

Topological entropy is a measure of the complexity of a topological space based on its persistence diagram. There are several ways to calculate topological entropy, but one of the most common is to use Shannon entropy over the distribution of intervals in the persistence diagram:

Obtain the interval distribution: Consider the birth-death intervals in the persistence diagram (the points on the diagram).

Calculate interval lengths: For each point in the diagram, calculate the length of the interval (death - birth).

Normalize lengths: Calculate the sum of all interval lengths and divide each length by this sum to obtain a probability distribution.

Calculate entropy: Using the Shannon entropy formula, calculate the entropy of the probability distribution of interval lengths.

$H = -\sum_{i \in I} p_i \log p_i$

where $p_i$ is the probability of interval length i.

This entropy provides a measure of the topological complexity of the point cloud, and can be used to compare different datasets or study the structure of a particular dataset.