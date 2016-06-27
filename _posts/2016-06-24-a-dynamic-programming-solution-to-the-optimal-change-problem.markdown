---
layout: post
title:  "A Dynamic Programming Solution To The Optimal Change Problem"
date:   2016-06-24 13:58:49 -0700
categories: dynamic programming algorithms
---

## Problem
In some circumstances – indeed, the majority of randomly chosen ones – devising optimal change for a given amount A and a set of denominations `{d1, d2, ...dn}` will not yield to a greedy algorithm, but requires dynamic programming. Give carefully worded pseudo-code for an algorithm that takes as input an amount and a set of denominations (such as coins or stamps) and generates on optimal collection of change (optimal in the sense that it has the fewest total number of elements (coins or stamps).

## Solution
Supposing the denominations are 1,6,10 and the amount is 24, my algorithm generates a 25 x 3 matrix (24 + the zero column) that stores the optimal coin counts for each amount from 0 to 24. Each column denoted by index `amt` (amount) is computed by looking at the `amt-coin` columns for each coin in denominations. So in the case of 24, the algorithm looks at columns (24-1), (24-6), and (24-10). For each of these three columns, the algorithm finds how many coins of the respective denomination need to be added to equal 24. The denomination with the lowest number of coins is chosen and the coin amounts are written to the matrix. My algorithm follows the dynamic programming methodology because it breaks the problem into overlapping sub-problems.

```python
denominations = [1,6,10]
amount = 24

# Counts is a matrix that stores the counts of each coin for each value in the range (0, amount)
# Counts = [amount][number of denominations]
counts = [[0 for x in range(len(denominations)) ] for x in range(amount+1)]

for amt in range(1,amount+1):
    # Set the minimum coin count to infinity for this amount
    min_sum = float('inf')
    # For each coin, look at the optimal coin counts for amt-coin.
    # Then check if using that coin to make up the difference from amt to (amt - coin) leads to using the fewest coins.
    for i in range(len(denominations)):
        coin = denominations[i]
        if denominations[i]<=amt:
            if ((amt-(amt-coin))) in denominations:
                sum=0
                for n in range(0,len(denominations)):
                    sum=(sum+counts[amt-coin][n])
                sum=sum+1
                if sum<min_sum:
                    min_sum=sum
                    optimal_coin = i
    temp = list(counts[amt-denominations[optimal_coin]])
    temp[optimal_coin]=temp[optimal_coin]+1
    counts[amt]=temp

print counts

print "The optimal change for ", amount, "is "
for i in range(len(denominations)):
    print counts[amount][i], " of coin ", denominations[i]
print "Total coins = ", min_sum
```

The output of this algorithm with amount = 24 and denominations = [1,6,10] is:

```
[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [0, 0, 1], [1, 0, 1],
[0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0], [0, 1, 1], [1, 1, 1],
[0, 3, 0], [1, 3, 0], [0, 0, 2], [1, 0, 2], [0, 2, 1], [1, 2, 1],
[0, 4, 0]]

The optimal change for 24 is
0  of coin  1
4  of coin  6
0  of coin  10
Total coins =  4
```

Here's a sketch I did when devising this solution. Amount = 12 and denominations = [1,6,10]:
![whiteboard](/img/1.jpg)
