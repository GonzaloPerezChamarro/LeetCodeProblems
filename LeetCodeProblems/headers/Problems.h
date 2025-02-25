
#pragma once

#include <vector>
#include <string>

#include "LeetCode_fwd.h"

using namespace std;

class Solution
{
public:
    /**
    * 1. Two Sum
    * 
    */
    vector<int> twoSum(vector<int>& nums, int target);

    /**
    * 2. Add Two Numbers
    *
    */
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

    /**
    * 2. Add Two Numbers Variant
    *
    */
    ListNode* addTwoNumbersVariant(ListNode* l1, ListNode* l2);

    /**
    * 7. Reverse Integer
    *
    */
    int reverse(int x);

    /**
    * 9. Palindrome Number
    *
    */
    bool isPalindrome(int x);

    /**
    * 12. Integer to Roman
    *
    */
    string intToRoman(int num);

    /**
    * 12. Integer to Roman Variant
    *
    */
    string intToRomanVariant(int num);

    /**
    * 13. Roman to Integer
    *
    */
    int romanToInt(string s);

    /**
    * 14. Longest Common Prefix
    *
    */
    string longestCommonPrefix(vector<string>& strs);

    /**
    * 14. Longest Common Prefix
    *
    */
    string longestCommonPrefixComplex(vector<string>& strs);

    /**
    * 17. Letter Combinations of a Phone Number
    *
    */
    vector<string> letterCombinations(string digits);

    /**
    * 17. Letter Combinations of a Phone Number
    *
    */
    vector<string> letterCombinationsVariant(string digits);

    /**
    * 20. Valid Parentheses
    *
    */
    bool isValid(string s);

    /**
    * 21. Merge Two Sorted Lists
    *
    */
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2);

    /**
    * 26. Remove Duplicates from Sorted Array
    *
    */
    int removeDuplicates(vector<int>& nums);

    /**
    * 27. Remove Element
    *
    */
    int removeElement(vector<int>& nums, int val);

    /**
    * 28. Find the Index of the First Occurrence in a String
    *
    */
    int strStr(string haystack, string needle);

    /**
    * 35. Search Insert Position
    *
    */
    int searchInsert(vector<int>& nums, int target);

    /**
    * 36. Valid Sudoku
    *
    */
    bool isValidSudoku(vector<vector<char>>& board);

    /**
    * 49. Group Anagrams
    *
    */
    vector<vector<string>> groupAnagrams(vector<string>& strs);

    /**
    * 53. Maximum Subarray
    *
    */
    int maxSubArray(vector<int>& nums);

    /**
    * 55. Jump Game
    *
    */
    bool canJump(vector<int>& nums);

    /**
    * 58. Length of Last Word
    *
    */
    int lengthOfLastWord(string s);

    /**
    * 66. Plus One
    *
    */
    vector<int> plusOne(vector<int>& digits);
    
    /**
    * 67. Add Binary
    *
    */
    string addBinary(string a, string b);

    /**
    * 69. Sqrt(x)
    *
    */
    int mySqrt(int x);

    /**
    * 70. Climbing Stairs
    *
    */
    int climbStairs(int n);

    /**
    * 80. Remove Duplicates from Sorted Array II
    *
    */
    int removeDuplicatesMedium(vector<int>& nums);

    /**
    * 88. Merge Sorted Array
    *
    */
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);

    /**
    * 100. Same Tree
    *
    */
    bool isSameTree(TreeNode* p, TreeNode* q);

    /**
    * 101. Symmetric Tree
    *
    */
    bool isSymmetric(TreeNode* root);

    /**
    * Aux function of -> 101. Symmetric Tree
    *
    */
    bool isMirrorTree(TreeNode* left, TreeNode* right);

    /**
    * 104. Maximum Depth of Binary Tree
    *
    */
    int maxDepth(TreeNode* root);

    /**
    * 105. Construct Binary Tree from Preorder and Inorder Traversal
    *
    */
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);

    /**
    * 108. Convert Sorted Array to Binary Search Tree
    *
    */
    TreeNode* sortedArrayToBST(vector<int>& nums);

    /**
    * 112. Path Sum
    *
    */
    bool hasPathSum(TreeNode* root, int targetSum);

    /**
    * 121. Best Time to Buy and Sell Stock
    *
    */
    int maxProfit(vector<int>& prices);


    /**
    * 121. Best Time to Buy and Sell Stock
    *
    */
    int maxProfitVariant(vector<int>& prices);

    /**
    * 122. Best Time to Buy and Sell Stock II
    *
    */
    int maxProfitMedium(vector<int>& prices);

    /**
    * 125. Valid Palindrome
    *
    */
    bool isPalindrome(string s);

    /**
    * 128. Longest Consecutive Sequence
    *
    */
    int longestConsecutive(vector<int>& nums);

    /**
    * 136. Single Number
    *
    */
    int singleNumber(vector<int>& nums);

    /**
    * 136. Single Number Variant
    *
    */
    int singleNumberVariant(vector<int>& nums);

    /**
    * 137. Single Number II
    *
    */
    int singleNumberMedium(vector<int>& nums);

    /**
    * 141. Linked List Cycle
    *
    */
    bool hasCycle(ListNode* head);

    /**
    * 167. Two Sum II - Input Array Is Sorted
    *
    */
    vector<int> twoSumMedium(vector<int>& numbers, int target);

    /**
    * 169. Majority Element
    *
    */
    int majorityElement(vector<int>& nums);

    /**
    * 189. Rotate Array
    *
    */
    void rotate(vector<int>& nums, int k);

    /**
    * 189. Rotate Array
    *
    */
    void rotateVariant(vector<int>& nums, int k);

    /**
    * 190. Reverse Bits
    *
    */
    uint32_t reverseBits(uint32_t n);

    /**
    * 191. Number of 1 Bits
    *
    */
    int hammingWeight(int n);

    /**
    * 191. Number of 1 Bits
    *
    */
    int hammingWeightVariant(int n);

    /**
    * 200. Number of Islands
    *
    */
    int numIslands(vector<vector<char>>& grid);

    /**
    * 202. Happy Number
    *
    */
    bool isHappy(int n);

    /**
    * 202. Happy Number
    *
    */
    bool isHappyFloyd(int n);

    /**
    * Aux function of --> 202. Happy Number
    *
    */
    int getSumatory(int n);

    /**
    * 205. Isomorphic Strings
    *
    */
    bool isIsomorphic(string s, string t);

    /**
    * 215. Kth Largest Element in an Array
    *
    */
    int findKthLargest(vector<int>& nums, int k);

    /**
    * 219. Contains Duplicate II
    *
    */
    bool containsNearbyDuplicate(vector<int>& nums, int k);

    /**
    * 222. Count Complete Tree Nodes
    *
    */
    int countNodes(TreeNode* root);

    /**
    * 226. Invert Binary Tree
    *
    */
    TreeNode* invertTree(TreeNode* root);

    /**
    * 242. Valid Anagram
    *
    */
    bool isAnagram(string s, string t);

    /**
    * 290. Word Pattern
    *
    */
    bool wordPattern(string pattern, string s);

    /**
    * 383. Ransom Note
    *
    */
    bool canConstruct(string ransomNote, string magazine);

    /**
    * 392. Is Subsequence
    *
    */
    bool isSubsequence(string s, string t);

    /**
    * 530. Minimum Absolute Difference in BST
    *
    */
    int getMinimumDifference(TreeNode* root);
    
    /**
    * 637. Average of Levels in Binary Tree
    *
    */
    vector<double> averageOfLevels(TreeNode* root);

    /**
    * 918. Maximum Sum Circular Subarray
    *
    */
    int maxSubarraySumCircular(vector<int>& nums);
};
