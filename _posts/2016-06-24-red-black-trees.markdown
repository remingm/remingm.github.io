---
layout: post
title:  "Efficient Red-Black Tree Augmentation"
date:   2016-06-24 18:01:49 -0700
categories: machine learning TensorFlow neural networks MLB baseball
---

## Problem
Consider storing a dynamic set *S* of numbers in a red-black tree. The typical operations supported by a red-black tree are Insert, Search and Delete but we will add another operation, namely, MinDiff, which gives the absolute value of the difference between the two closest numbers in *S*. For example if *S = {4,18,23,34,49,60,73,84,99}* then the MinDiff function would return 5 since *|23 - 18| = 5* and the absolute value of the difference between any other two numbers in the set is no smaller.
Show how to augment a red-black tree to support this operation efficiently while maintaining the *O(lg n)* running time for Insert, Search and Delete. Also, show how to output the values of two numbers that created the MinDiff. For the example above you would output 23 and 18.

## Solution
Each node in the red-black tree will be augmented with four fields:

1. The minimum of either `node.value - node.left.max` (maximum value in left sub-tree) or `node.value - node.right.min` (minimum value in right sub-tree). This is the minimum difference of the current node with any node in its sub-tree.
2. Maximum value in sub-tree (`node.right.max`). If no right child, put the value of the current node.
3. Minimum value in sub-tree (`node.left.min`). If no left child, put the value of the current node.
4. Minimum difference in sub-tree. This differs from the first field in that this is the minimum difference between any two nodes in the sub-tree, not just the current node. This is the minimum of the following three fields: `node.left.field4, node.right.field4, node.field1`.

*O(lg n)* running time for Insert, Search and Delete is maintained because theorem 14.1 (1) from *Introduction to Algorithms* by Cormen, is valid for a red-black tree augmented with the above fields. This is because all of the augmented fields of a node can be determined from examining only the augmented fields of the immediate left child and/or right child. It is only necessary to look at the largest value in the left sub-tree and the smallest value in the right sub-tree when determining minimum difference with the parent node, as these values will be the closest in value to the parent node. After any insert, search, or delete operations, the root node will be augmented with the minimum di↵erence in the tree in field 4.

Finding the values of the two numbers that created the MinDiff can be done in *O(lgn)* time: Starting at the root, if `node.field4 = node.left.field4`, go left, otherwise go right. Repeat until `node.field1 = root.field4`. At this point, the current node is one of the values of the minimum difference, and the other is either `node.right.field2` or `node.left.field3`.

---
1.

>Theorem 14.1 (Augmenting a red-black tree)
Let f be an attribute that augments a red-black tree T of n nodes, and suppose that the value of f for each node x depends on only the information in nodes x, x:left, and x:right, possibly including x:left:f and x:right:f. Then, we can maintain the values of f in all nodes of T during insertion and deletion without asymptotically affecting the O.lg n/ performance of these operations.
