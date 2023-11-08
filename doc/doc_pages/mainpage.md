@mainpage Overview

## IVSparse (Index and Value Sparse) Library

[Github Repo](https://github.com/Seth-Wolfgang/IVSparse)

One of the most popular ways of storing and using sparse data is in Compressed Sparse Column (CSC) format. As a general-purpose sparse data storage format, CSC provides no mechanism to take advantage of redundant data. This data, found in many areas such as genomics and machine learning, can end up prohibitively expensive to store in CSC format, where each value is stored explicitly. Taking advantage of redundant data to compress, we propose two original formats that build upon CSC, Value Compressed Sparse Column (VCSC) as well as Index and Value Compressed Sparse Column (IVCSC). These formats compress values by only storing the unique values in each column, minimizing the necessary information for reading values and the amount of values stored. IVCSC takes compression a step further by positive-delta encoding and byte-packing the indices of each unique value, allowing indices to take as little as one byte to store. Our testing collected data of our formats compared to popular alternatives such as CSC, which benchmarking shows IVCSC can compress down to 20% the size, with a reasonable computational trade-off, and VCSC can compress down to 35% but performs faster. These two compression formats offer a novel solution to storing and using redundant sparse data at a minimal computational cost.

### What is IVSparse?

IVSparse is a library with two new ways to store sparse data much like compressed sparse column (CSC) or coordinate format (COO). These two new compression formats are called Value Compressed Sparse Column (VCSC) and Index and Value Compressed Sparse Column (IVCSC). These two formats are meant to take advantage of highly redundant data and compress it by value or by index and value respectively, without losing too much traversal speed.
___

### How does VCSC and IVCSC Compress?

There are a few different ways in which IVSparse compresses data. The first is redundancy, by only storing unique values in a column much data can be saved especially for larger sized values like doubles. The second is Positive Delta Encoding, which is a process which we apply to the indices of a unique value in a column which encodes the distance between each index. The third way is through bytepacking, which is when we cast each unqiue value's indices to the smallest data type that doesn't lose precision. VCSC uses the value compression technique and IVCSC uses the value compression as well as postive delta encoding and bytepacking to compress indices. 

#### Redundancy

It's easy to see how redundancy can be taken advantage of when looking at CSC format. In this format each value and index must be listed once no matter the data, with a set of pointers being used to delimit columns. However in VCSC and IVCSC if a column has only a single value, it's only stored once and then all of the indices that value is located will be associated with it. This means that in the simplest case of a vector of 50 ones, CSC needs to store 100 values where as VCSC and IVCSC needs only to store 51, the 1 (value) and all the indices where 1 appears. 

Therefore in datasets that are highly redundant we mananges to not explicitly store a lot of values, resulting in good compression ratios for this data. It is worth being said however that data that is almost completely unique however will cause this pendulum to swing in the other direction, causing worse compression in the worst case scenario since VCSC and IVCSC has more overhead to organize the data than CSC. 

#### Positive-Delta Endoding

We use this to take the indices inside of a unique value and encode the distance between them. While this doesn't cause us to store any less values, it can take larger values and make them smaller. Such as for a 10,000 x 10,000 matrix near the end of a column the indices could be [..., 9,973, 9,979, 9,981, 9,991], each of these values requires a 16 byte data type to store whereas if we positive delta encode them into [..., 3, 6, 2, 10] the values get much smaller and could potentially fit into a smaller data type. 

It should also be noted that this does make the data harder to traverse making it a tradeoff that isn't always worth it and will make data traversal slower, espeically if the data doesn't lend itself to the advantages of positive delta encoding by being incredibly sparse or with distances very far apart.

#### Byte-Packing

This is a process that takes all of the indices of a unique value, finds the maximum value after positive delta encoding, and casts the indices to the smallest data type that doesn't lose precision. Such as if a user has a IVCSC matrix with indices stored in `unsigned long long int` yet no number exceedes 255. In this situation each unique value's fiber's indices would most likely be cast to a unsigned short int saving 7 bytes per index. This is great for isolating outlier data from enforcing a large data type and in combination with positive delta encoding can result in even very large matrices being heavily conmpressed with very little wasted space in the indices. 

As a side note, this as well makes the data somewhat more difficult to work with causing some degree of complication and slowdowns for data traversal but is often very much worth the savings. 

___

### Compression Formats Explained

There are currently 3 compression formats supported by IVSparse which are presented in order of compression below.

*Compressed Sparse Column (CSC):*

This is simply just the CSC matrix format. This is helpful for transitioning between deeper compression levels and back to a more workable format as many times CSC is far faster for certain algorithms. This also helps increase interoperability with other libraries and within IVSparse. CSC also works better for matrices with mostly unique values and has the fastest traversal speeds.

*Value Compressed Sparse Column (VCSC):*

VCSC is a derivative of CSC meant to implement value compression on redundant values. This is done by storing only the unique values of each column/row and the number of occurances of each value. This means there are three arrays per outer dimension, the values, the counts, and the inner indices of the values. This format has fast scalar operations and is somewhat fast to traverse due to the counts array. This format also needs to have a fair amount of redundant values to benefit over CSC and is not recommended for data that is mostly unique.

*Index and Value Compressed Sparse Column (IVCSC):*

IVCSC is the format meant to focus primarily on compressing. This is done by compressing both indices as well as values. Firstly, IVCSC stores each column in its own contiguous block of memory. Inside each column's memory is a series of runs. Runs are associated with each unique value in the column. Runs are sorted by unique value in ascending order. The format of a run is the value, the width of the follwing indices, the indices associated with the unique value positive delta encoded, and finally a delimiter to signify the end of a run. IVCSC uses both index and value compression to achieve the highest compression ratio. The values are compressed by only storing the unique values of each column and the indices for each individual run are positive delta encoded and then bytepacked into the smallest usable size. This format is the slowest to traverse but has the highest compression ratio, that isn't to say one can't do operations with IVCSC they just won't be quite as fast as a CSC matrix.