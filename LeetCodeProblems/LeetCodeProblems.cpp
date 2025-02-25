
// LeetCodeProblems.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "Problems.h"
#include "LeetCode_fwd.h"


int main()
{
    std::cout << "--- Leet Code Problems! --- \n";
    std::cout << "------ Gonzalo Perez ------ \n\n";

    /// ------------------------------------------------------------------- ///

    // Vector example
    std::vector<int> v1 = { 1, 2, 3, 4, 5, 6 };

    // String example
    std::string str1 = "abba";

    // Tree node example
    TreeNode* treeNode6 = new TreeNode(3);
    TreeNode* treeNode5 = new TreeNode(4);
    TreeNode* treeNode4 = new TreeNode(4);
    TreeNode* treeNode3 = new TreeNode(3);
    TreeNode* treeNode2 = new TreeNode(2, treeNode5, treeNode6);
    TreeNode* treeNode1 = new TreeNode(2, treeNode3, treeNode4);
    TreeNode* treeRoot = new TreeNode(2, treeNode1, treeNode2);

    // Linked list 1
    ListNode* nodeD = new ListNode(9);
    ListNode* nodeC = new ListNode(4, nodeD);
    ListNode* nodeB = new ListNode(6, nodeC);
    ListNode* nodeA = new ListNode(5, nodeB);

    // Linked list 2
    ListNode* node3 = new ListNode(9);
    ListNode* node2 = new ListNode(4, node3);
    ListNode* node1 = new ListNode(2, node2);

    /// ------------------------------------------------------------------- ///

    // Solution class containts Leet Code problems
    // Call Solution.problemName(...);
    Solution solution;

    std::cout << "Problem --> isPalindrome(abba) [true]: " << solution.isPalindrome(str1) << "\n";

    std::cin.get();
}
