
#include "Problems.h"

#include <unordered_map>
#include <algorithm>
#include <math.h>
#include <unordered_set>
#include <queue>
#include <functional>
#include <bitset>
#include <array>

vector<int> Solution::twoSum(vector<int>& nums, int target)
{
    const int size = nums.size();

    int index;
    for (index = 0; index < size - 1; ++index)
    {
        for (int j = index + 1; j < size; ++j)
        {
            if (nums[index] + nums[j] == target)
            {
                return { index, index + 1 };
            }
        }
    }

    return { index - 1, index };
}

vector<int> Solution::twoSumMedium(vector<int>& numbers, int target)
{
    const int size = numbers.size();
    int i = 0, j = size - 1;

    while (i <= j)
    {
        const int sum = numbers[i] + numbers[j];
        if (sum == target)
        {
            return { i + 1, j + 1 };
        }

        if (sum < target)
        {
            ++i;
        }
        else
        {
            --j;
        }
    }

    return vector<int>();
}

int Solution::reverse(int x)
{
    if (x == 0)
    {
        return 0;
    }

    int reversed = 0;

    while (x != 0)
    {
        if (reversed > INT_MAX / 10 || reversed < INT_MIN / 10)
        {
            return 0;
        }
        reversed *= 10;
        reversed += (x % 10);
        x /= 10;
    }

    if (reversed > INT_MAX || reversed < INT_MIN)
    {
        return 0;
    }

    return reversed;
}

bool Solution::isPalindrome(int x)
{
    if (x < 0) return false;

    long long reverse = 0;
    long long aux = x;

    while (aux != 0)
    {
        reverse *= 10;
        reverse += aux % 10;

        aux /= 10;
    }

    return x == reverse;
}

int Solution::romanToInt(string s)
{
    // Symbol, value
    static std::unordered_map<char, int> romanTable =
    { {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000} };

    int intValue = 0;

    int lastValue = 0;
    for (int i = s.length() - 1; i >= 0; --i)
    {
        const int value = romanTable[s[i]];

        if (value < lastValue)
        {
            intValue += lastValue - value;
            lastValue = 0;
        }
        else
        {
            intValue += lastValue;
            lastValue = value;
        }
    }

    intValue += lastValue;

    return intValue;
}

string Solution::intToRoman(int num)
{
    // This solution compare with the values directly

    int values[] = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
    std::string symbols[] = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };

    std::string result;
    for (size_t i = 0; i < 13; ++i) 
    {
        while (num >= values[i]) 
        {
            num -= values[i];
            result += symbols[i];
        }
    }

    return result;
}

string Solution::intToRomanVariant(int num)
{
    // This solution compose the solution with the basic symbols

    static std::unordered_map<int, char> romanTable =
    { {1, 'I'},
        {5, 'V'},
        {10, 'X'},
        {50, 'L'},
        {100, 'C'},
        {500, 'D'},
        {1000, 'M'} };

    string result;

    int count = 0;

    while (num > 0)
    {
        int current = num % 10;
        num /= 10;

        const int pow = static_cast<int>(std::pow(10, count));

        string aux;
        if (current == 9)
        {
            aux.push_back(romanTable[10 * pow]);
            aux.push_back(romanTable[1 * pow]);
            current = 0;
        }
        else
        {
            if (current >= 5)
            {
                aux.push_back(romanTable[5 * pow]);
                current -= 5;
            }

            if (current > 3)
            {
                aux.push_back(romanTable[5 * pow]);
                aux.push_back(romanTable[1 * pow]);
            }
            else
            {
                const char one = romanTable[1 * pow];
                while (current > 0)
                {
                    aux.push_back(one);
                    --current;
                }

                std::reverse(aux.begin(), aux.end());
            }
        }

        result.append(aux);
        ++count;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

string Solution::longestCommonPrefixComplex(vector<string>& strs)
{
    if (strs.size() == 1) return strs[0];

    string resultPrefix;

    int resultCount = 0;

    // for each string
    for (size_t i = 0; i < strs.size() - 1; ++i)
    {
        string& current = strs[i];

        string prefix;
        int count = 0;

        // compare with next strings
        for (size_t j = i + 1; j < strs.size(); ++j)
        {
            string& aux = strs[j];

            const size_t shortestSize = current.size() < aux.size() ? current.size() : aux.size();
            string commonPrefix;

            // Get common prefix
            for (size_t k = 0; k < shortestSize; ++k)
            {
                if (current[k] != aux[k])
                {
                    break;
                }

                commonPrefix.push_back(current[k]);
            }

            if (commonPrefix.size() > 0)
            {
                int compResult = prefix.compare(0, prefix.size(), commonPrefix);
                if (count == 0 || compResult == 0 || (commonPrefix.size() < prefix.size() && compResult > 0))
                {
                    prefix = commonPrefix;
                    ++count;
                }
            }
        }

        if (count > 0 && (count > resultCount || count == resultCount && prefix.size() > resultPrefix.size()))
        {
            resultPrefix = prefix;
            resultCount = count;
        }
    }

    return resultPrefix;
}

string Solution::longestCommonPrefix(vector<string>& strs)
{
    const size_t size = strs.size();
    const string& first = strs[0];

    std::sort(strs.begin(), strs.end());

    string result = first;

    for (size_t i = 1; i < size; ++i)
    {
        const string& current = strs[i];

        string commonPrefix;
        for (size_t j = 0; j < current.size() && j < first.size() && commonPrefix.size() < result.size(); ++j)
        {
            const auto& character = current[j];
            if (first[j] != character)
            {
                break;
            }

            commonPrefix.push_back(character);
        }

        result = commonPrefix;
    }

    return result;
}

bool Solution::isValid(string s)
{
    static unordered_map<char, char> parentheses = { {'(', ')'}, {'{', '}'} , {'[', ']'} };

    const size_t size = s.size();

    if (size % 2 != 0) return false;

    std::vector<char> openParentheses;

    for (size_t i = 0; i < size; ++i)
    {
        char& c = s[i];

        if (parentheses.contains(c))
        {
            openParentheses.emplace_back(c);
        }
        else
        {
            if (openParentheses.size() == 0 || parentheses[openParentheses.back()] != c)
            {
                return false;
            }

            openParentheses.pop_back();
        }
    }

    return openParentheses.size() == 0;
}

ListNode* Solution::mergeTwoLists(ListNode* list1, ListNode* list2)
{
    ListNode* result = nullptr;
    ListNode* current = nullptr;

    ListNode* pt1 = list1;
    ListNode* pt2 = list2;

    while (pt1 != nullptr || pt2 != nullptr)
    {
        int val1 = 101, val2 = 101;

        if (pt1 != nullptr)
        {
            val1 = pt1->val;
        }

        if (pt2 != nullptr)
        {
            val2 = pt2->val;
        }

        ListNode** toCopy = val1 <= val2 ? &pt1 : &pt2;
        const int value = (*toCopy)->val;

        if (current != nullptr)
        {
            current->next = new ListNode(value);
            current = current->next;
        }
        else
        {
            current = new ListNode(value);
            result = current;
        }

        *toCopy = (*toCopy)->next;
    }

    return result;
}

int Solution::removeDuplicates(vector<int>& nums)
{
    const size_t size = nums.size();

    int deletedNumbers = 0;
    int prevValue = INT_MIN;
    for (size_t i = 0; i < size; ++i)
    {
        const int value = nums[i];
        if (value == prevValue)
        {
            nums[i] = INT_MAX;
            ++deletedNumbers;
        }
        else
        {
            prevValue = value;
        }
    }

    sort(nums.begin(), nums.end());

    return size - deletedNumbers;
}

int Solution::removeDuplicatesMedium(vector<int>& nums)
{
    const size_t size = nums.size();
    if (size <= 2) return size;

    int j = 2;
    for (size_t i = 2; i < size; ++i)
    {
        if (nums[i - 1] != nums[i]) 
        {
            nums[j] = nums[i];
            j++;
        }
        else if (nums[j - 1] != nums[j - 2]) 
        {
            nums[j] = nums[i];
            j++;
        }
    }

    return j;
}

int Solution::removeElement(vector<int>& nums, int val)
{
    int currentSize = nums.size();

    if (currentSize == 1 && nums[0] == val)
    {
        return 0;
    }

    for (int i = 0; i < currentSize; ++i)
    {
        if (nums[i] == val)
        {
            const int aux = nums[currentSize - 1];
            nums[currentSize - 1] = val;
            nums[i] = aux;
            --currentSize;
            --i;
        }
    }

    return currentSize;
}

int Solution::strStr(string haystack, string needle)
{
    return haystack.find(needle, 0);
}

int Solution::searchInsert(vector<int>& nums, int target)
{
    size_t begin = 0;
    size_t end = nums.size() - 1;

    while (begin < end)
    {
        const size_t middle = begin + static_cast<size_t>(((end - begin) * 0.5f));

        if (begin < middle && nums[middle] < target)
        {
            begin = middle;
        }
        else if (nums[end] < target)
        {
            begin = end;
        }
        else
        {
            end = middle;
        }
    }

    return nums[begin] >= target ? begin : begin + 1;
}

void Solution::merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
{
    for (int i = m; i < m + n; ++i)
    {
        nums1[i] = nums2[i - m];
    }

    std::sort(nums1.begin(), nums1.end());
}

int Solution::majorityElement(vector<int>& nums)
{
    std::unordered_map<int, size_t> amount;

    for (const auto& number : nums)
    {
        ++amount[number];
    }

    int majorityElement = -1;
    size_t bestAmount = 0;
    for (const auto& entryAmount : amount)
    {
        if (bestAmount < entryAmount.second)
        {
            majorityElement = entryAmount.first;
            bestAmount = entryAmount.second;
        }
    }

    return majorityElement;
}

int Solution::maxProfit(vector<int>& prices)
{
    int profit = 0;

    int first = prices[0];
    int second = first;
    int lower = first;

    for (size_t i = 1; i < prices.size(); ++i)
    {
        const int price = prices[i];

        if (price > second)
        {
            second = price;
            first = lower;
            profit = second - first;
        }
        else if (price < lower)
        {
            lower = price;

            if (profit == 0)
            {
                first = second = price;
            }
        }
        else if (price - lower > profit)
        {
            first = lower;
            second = price;
            profit = second - first;
        }
    }

    return profit;
}

int Solution::maxProfitVariant(vector<int>& prices)
{
    int profit = 0;
    int minEle = INT_MAX;

    for (int price : prices) 
    {
        minEle = min(minEle, price);
        profit = max(profit, price - minEle);
    }

    return profit;
}

int Solution::maxProfitMedium(vector<int>& prices)
{
    int profit = 0;

    for (size_t i = 1; i < prices.size(); ++i)
    {
        if (prices[i] > prices[i - 1])
        {
            profit += prices[i] - prices[i - 1];
        }
    }

    return profit;
}

int Solution::lengthOfLastWord(string s)
{
    int count = 0;

    for (int i = s.size() - 1; i >= 0; --i)
    {
        if (s[i] != ' ')
        {
            ++count;
        }
        else if (count > 0)
        {
            break;
        }
    }

    return count;
}

bool Solution::isPalindrome(string s)
{
    string clearString;
    clearString.reserve(s.capacity());

    for (auto c : s)
    {
        if (std::isalnum(c))
        {
            clearString.push_back(std::tolower(c));
        }
    }

    for (int i = 0, j = clearString.size() - 1; i < j; ++i, --j)
    {
        if (clearString[i] != clearString[j])
        {
            return false;
        }
    }

    return true;
}

bool Solution::isSubsequence(string s, string t)
{
    const size_t sSize = s.size();
    const size_t tSize = t.size();

    size_t found = 0;

    for (size_t i = 0; i < tSize && (tSize - i >= sSize - found); ++i)
    {
        if (t[i] == s[found])
        {
            ++found;
        }
    }

    return found == s.size();
}

bool Solution::canConstruct(string ransomNote, string magazine)
{
    size_t found = 0;
    for (size_t i = 0; i < ransomNote.size() && found == i; ++i)
    {
        const char current = ransomNote[i];

        for (size_t j = 0; j < magazine.size() - found; ++j)
        {
            if (current == magazine[j])
            {
                ++found;
                magazine[j] = magazine[magazine.size() - found];
                magazine[magazine.size() - found] = current;
                break;
            }
        }
    }

    return ransomNote.size() == found;
}

bool Solution::isIsomorphic(string s, string t)
{
    if (s.length() != t.length()) return false;

    vector<int> indexS(200, 0);
    vector<int> indexT(200, 0);

    for (size_t i = 0; i < s.length(); i++) 
    {
        if (indexS[s[i]] != indexT[t[i]])
        {
            return false;
        }

        indexS[s[i]] = i + 1;
        indexT[t[i]] = i + 1;
    }

    return true;
}

bool Solution::wordPattern(string pattern, string s)
{
    const size_t patternSize = pattern.size();
    const size_t sSize = s.size();

    std::unordered_map<string, char> mappingToS;
    std::unordered_map<char, string> mappingToPattern;

    size_t wordIndex = 0;
    for (size_t i = 0; i < patternSize; ++i)
    {
        const char c = pattern[i];
        string word;

        while (wordIndex < sSize)
        {
            if (s[wordIndex] == ' ')
            {
                break;
            }

            word.push_back(s[wordIndex]);
            ++wordIndex;
        }

        if (word.empty())
        {
            return false;
        }

        if ((mappingToPattern.contains(c) && mappingToPattern[c] != word)
            || (mappingToS.contains(word) && mappingToS[word] != c))
        {
            return false;
        }

        mappingToPattern[c] = word;
        mappingToS[word] = c;

        ++wordIndex;
    }

    return wordIndex >= sSize;
}

bool Solution::isAnagram(string s, string t)
{
    if (s.size() != t.size())
    {
        return false;
    }

    unordered_map<char, int> mS;
    unordered_map<char, int> mT;

    for (size_t i = 0; i < s.size(); ++i)
    {
        mS[s[i]]++;
        mT[t[i]]++;
    }

    return mS == mT;
}

bool Solution::isHappy(int n)
{
    std::unordered_set<int> foundNumbers;
    while (n != 1)
    {
        int sum = 0;
        int digit = 0;
        while (n > 0)
        {
            digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }

        n = sum;

        if (foundNumbers.contains(n))
        {
            break;
        }

        foundNumbers.emplace(n);
    }

    return n == 1;
}

int Solution::getSumatory(int n)
{
    // isHappyFloyd auxiliar function

    int sum = 0;
    while (n > 0)
    {
        int digit = n % 10;
        sum += digit * digit;
        n /= 10;
    }
    return sum;
}

bool Solution::isHappyFloyd(int n)
{
    // Using Floyd's Algorithm

    int slow = n;
    int fast = n;

    do
    {
        slow = getSumatory(slow);
        fast = getSumatory(getSumatory(fast));
    } 
    while (slow != fast);

    return slow == 1;
}

bool Solution::hasCycle(ListNode* head)
{
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != NULL && fast->next != NULL)
    {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast)
        {
            return true;
        }
    }

    return false;
}

int Solution::maxDepth(TreeNode* root)
{
    if (root == nullptr)
    {
        return 0;
    }

    return 1 + std::max(maxDepth(root->left), maxDepth(root->right));
}

bool Solution::isSameTree(TreeNode* p, TreeNode* q)
{
    if (p == nullptr || q == nullptr)
    {
        return p == q;
    }

    if (p->val != q->val)
    {
        return false;
    }

    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

vector<double> Solution::averageOfLevels(TreeNode* root)
{
    std::vector<double> averages;
    if (!root) return averages;

    // Breadth First Search (BFS)
    std::queue<TreeNode*> q;
    q.push(root);

    while (!q.empty())
    {
        int levelSize = q.size();
        double levelSum = 0;

        for (int i = 0; i < levelSize; ++i) 
        {
            TreeNode* node = q.front();
            q.pop();
            levelSum += node->val;

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        averages.push_back(levelSum / levelSize);
    }

    return averages;
}

TreeNode* Solution::invertTree(TreeNode* root)
{
    if (root != nullptr)
    {
        TreeNode* aux = invertTree(root->left);
        root->left = invertTree(root->right);
        root->right = aux;
    }

    return root;
}

bool Solution::isMirrorTree(TreeNode* left, TreeNode* right)
{
    // isSymetric aux function

    if (left == nullptr || right == nullptr)
    {
        return left == right;
    }

    return (left->val == right->val)
        && (isMirrorTree(left->left, right->right))
        && (isMirrorTree(left->right, right->left));
}

bool Solution::isSymmetric(TreeNode* root)
{
    if (root == nullptr) return true;

    return isMirrorTree(root->left, root->right);
}

bool Solution::hasPathSum(TreeNode* root, int targetSum)
{
    if (root == nullptr)
    {
        return false;
    }

    targetSum -= root->val;
    if (targetSum == 0 && (root->left == nullptr && root->right == nullptr))
    {
        return true;
    }

    return hasPathSum(root->left, targetSum) || hasPathSum(root->right, targetSum);
}

int Solution::countNodes(TreeNode* root)
{
    if (root == nullptr)
    {
        return 0;
    }

    return countNodes(root->left) + countNodes(root->right) + 1;
}

int Solution::getMinimumDifference(TreeNode* root)
{
    // It works with a Binary Search Tree (BST)

    int minDiff = INT_MAX;
    int prev = -1;

    // Depth First Search (DFS)
    std::function<void(TreeNode*)> inOrder = [&](TreeNode* node)
    {
        if (!node) return;

        inOrder(node->left);

        if (prev != -1)
        {
            minDiff = min(minDiff, node->val - prev);
        }
        prev = node->val;

        inOrder(node->right);
    };

    inOrder(root);
    return minDiff;
}

TreeNode* Solution::sortedArrayToBST(vector<int>& nums)
{
    std::function<TreeNode* (int, int)> InOrder = [&](int begin, int end) -> TreeNode*
    {
        if (begin > end)
        {
            return nullptr;
        }

        int mid = begin + ((end - begin) / 2);
        TreeNode* root = new TreeNode(nums[mid]);

        root->left = InOrder(begin, mid - 1);
        root->right = InOrder(mid + 1, end);

        return root;
    };

    return InOrder(0, nums.size() - 1);
}

string Solution::addBinary(string a, string b)
{
    string result;

    int i = a.size() - 1;
    int j = b.size() - 1;

    int carry = 0;

    while (i >= 0 || j >= 0 || carry)
    {
        int sum = carry;

        if (i >= 0)
        {
            sum += a[i] - '0';
            --i;
        }

        if (j >= 0)
        {
            sum += b[j] - '0';
            --j;
        }

        result.push_back((sum % 2) + '0');
        carry = sum / 2;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

uint32_t Solution::reverseBits(uint32_t n)
{
    uint32_t result = 0;

    for (size_t i = 0; i < 32; ++i)
    {
        if (n & (1 << i))
        {
            result |= 1 << (31 - i);
        }
    }

    return result;
}

int Solution::hammingWeight(int n)
{
    // Manual counting 
    int hammingWeight = 0;

    for (size_t i = 0; i < 8 * sizeof(n); ++i)
    {
        if (n & (1 << i))
        {
            ++hammingWeight;
        }
    }

    return hammingWeight;
}

int Solution::hammingWeightVariant(int n)
{
    // Using a std::bitset<>
    bitset<32> b(n);
    return b.count();
}

int Solution::singleNumber(vector<int>& nums)
{
    // Using bit manipulation
    int result = 0;
    for (auto u : nums) 
    {
        result ^= u;
    }

    return result;
}

int Solution::singleNumberVariant(vector<int>& nums)
{
    // Sorting and searching the single number

    std::sort(nums.begin(), nums.end());

    for (size_t i = 0; i < nums.size(); i += 2)
    {
        if (i == nums.size() - 1 || nums[i] != nums[i + 1])
        {
            return nums[i];
        }
    }

    return INT_MIN;
}

int Solution::singleNumberMedium(vector<int>& nums)
{
    int once = 0, twice = 0;

    for (auto num : nums)
    {
        twice |= once & num;
        once ^= num;

        const int thrice = once & twice;

        once &= ~thrice;
        twice &= ~thrice;
    }

    return once;
}

vector<int> Solution::plusOne(vector<int>& digits)
{
    const size_t size = digits.size();
    std::vector<int> result(size, 0);

    int plus = 1;
    for (int i = size - 1; i >= 0; --i)
    {
        const int sum = digits[i] + plus;
        if (sum < 10)
        {
            result[i] = sum;
            plus = 0;
        }
        else
        {
            result[i] = 0;
            plus = 1;
        }
    }

    if (plus > 0)
    {
        result.insert(result.begin(), plus);
    }

    return result;
}

int Solution::mySqrt(int x)
{
    // binary search

    int left = 0;
    int right = x;

    while (left <= right)
    {
        long mid = (left + right) / 2;

        if (mid * mid < x)
        {
            left = mid + 1;
        }
        else if (mid * mid > x)
        {
            right = mid - 1;
        }
        else
        {
            return mid;
        }
    }

    return right;
}

int Solution::climbStairs(int n)
{
    if (n == 1) return 1;

    int onePrev = 1;
    int twoPrev = 1;
    int result = 0;

    for (int i = 2; i <= n; ++i)
    {
        result = onePrev + twoPrev;
        twoPrev = onePrev;
        onePrev = result;
    }

    return result;
}

void Solution::rotate(vector<int>& nums, int k)
{
    const int size = nums.size();
    if (k == 0 || size == 1 || k == size) return;

    if (k > size)
    {
        k = k % size;
    }

    vector<int> aux(nums.end() - k, nums.end());

    for (int i = size - k - 1, j = size - 1; j >= k; --i, --j)
    {
        nums[j] = nums[i];
    }

    for (int i = 0; i < k; ++i)
    {
        nums[i] = aux[i];
    }
}

void Solution::rotateVariant(vector<int>& nums, int k)
{
    const int size = nums.size();
    if (k > size) k %= size;

    std::reverse(nums.begin(), nums.begin() + (size - k));
    std::reverse(nums.begin() + (size - k), nums.end());
    std::reverse(nums.begin(), nums.end());
}

bool Solution::isValidSudoku(vector<vector<char>>& board)
{
    const size_t boardSize = board.size();

    int row[9] = { 0 }, col[9] = { 0 }, boxes[9] = { 0 };

    for (size_t i = 0; i < boardSize; ++i)
    {
        for (size_t j = 0; j < boardSize; ++j)
        {
            if (board[i][j] == '.') continue;

            const int num = board[i][j] - '1';  // Converting (char) '1'-'9' to (int) 0-8
            const int mask = 1 << num;
            const size_t boxIndex = ((i / 3) * 3) + (j / 3);

            if (row[i] & mask || col[j] & mask || boxes[boxIndex] & mask)
            {
                return false;
            }

            row[i] |= mask;
            col[j] |= mask;
            boxes[boxIndex] |= mask;
        }
    }

    return true;
}

bool Solution::containsNearbyDuplicate(vector<int>& nums, int k)
{
    const int size = nums.size();

    std::unordered_map<int, int> found;
    found.reserve(size);

    for (int i = 0; i < size; ++i)
    {
        const int number = nums[i];

        if (found.contains(number) && abs(found[number] - i) <= k)
        {
            return true;
        }

        found[number] = i;
    }


    return false;
}

vector<vector<string>> Solution::groupAnagrams(vector<string>& strs)
{
    std::unordered_map<std::string, vector<string>> found;

    for (const string& str : strs)
    {
        string copy = str;
        sort(copy.begin(), copy.end());

        found[copy].emplace_back(str);
    }

    vector<vector<string>> result;
    result.reserve(found.size());

    for (auto& list : found)
    {
        result.emplace_back(std::move(list.second));
    }

    return result;
}

int Solution::longestConsecutive(vector<int>& nums)
{
    if (nums.size() == 0) return 0;

    sort(nums.begin(), nums.end());

    int longest = 1;
    int current = 1;

    for (size_t i = 1; i < nums.size(); ++i)
    {
        if (nums[i] == nums[i - 1] + 1)
        {
            ++current;
        }
        else if (nums[i] != nums[i - 1])
        {
            if (current > longest)
            {
                longest = current;
            }
            current = 1;
        }

    }

    return max(longest, current);
}

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2)
{
    ListNode* root = new ListNode();
    ListNode* current = root;

    int plus = 0;
    while (l1 != nullptr || l2 != nullptr || plus > 0)
    {
        const int number1 = l1 != nullptr ? l1->val : 0;
        const int number2 = l2 != nullptr ? l2->val : 0;

        int result = number1 + number2 + plus;
        plus = 0;

        if (result >= 10)
        {
            result -= 10;
            ++plus;
        }

        current->val = result;

        l1 = l1 != nullptr ? l1->next : nullptr;
        l2 = l2 != nullptr ? l2->next : nullptr;

        if (l1 != nullptr || l2 != nullptr || plus > 0)
        {
            current->next = new ListNode();
            current = current->next;
        }
    }

    return root;
}

ListNode* Solution::addTwoNumbersVariant(ListNode* l1, ListNode* l2)
{
    // This is a variant of the solution that only accepts numbers <= INT_MAX
    auto getNumber = [](ListNode* node)
    {
        ListNode* current = node;
        int number = 0;
        int count = 0;

        while (current != nullptr)
        {
            number += current->val * static_cast<int>(pow(10, count));
            current = current->next;
            ++count;
        }

        return number;
    };

    const int number1 = getNumber(l1);
    const int number2 = getNumber(l2);

    int result = number1 + number2;

    ListNode* root = new ListNode();
    ListNode* aux = root;

    while (result > 0)
    {
        aux->val = result % 10;
        result /= 10;

        if (result > 0)
        {
            aux->next = new ListNode();
            aux = aux->next;
        }
    }

    return root;
}

TreeNode* Solution::buildTree(vector<int>& preorder, vector<int>& inorder)
{
    const int size = inorder.size();

    std::unordered_map<int, int> inOrderIndices;

    for (int i = 0; i < size; ++i)
    {
        inOrderIndices[inorder[i]] = i;
    }

    std::function<TreeNode* (int, int, int, int)> constructTree = [&](int preStart, int preEnd, int inStart, int inEnd) -> TreeNode*
    {
        if (preStart > preEnd || inStart > inEnd)
        {
            return nullptr;
        }

        TreeNode* root = new TreeNode();
        root->val = preorder[preStart];

        int rootIndex = inOrderIndices[root->val];
        int leftTreeSize = rootIndex - inStart;

        root->left = constructTree(preStart + 1, preStart + leftTreeSize, inStart, rootIndex - 1);
        root->right = constructTree(preStart + leftTreeSize + 1, preEnd, rootIndex + 1, inEnd);

        return root;
    };

    return constructTree(0, size - 1, 0, size - 1);
}

int Solution::numIslands(vector<vector<char>>& grid)
{
    static const char LAND = '1';
    static const char WATER = '0';

    int count = 0;

    std::function<void(size_t, size_t)> findNeighborLand = [&](size_t i, size_t j)
    {
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[i].size())
        {
            return;
        }

        if (grid[i][j] == WATER)
        {
            return;
        }

        grid[i][j] = WATER;

        findNeighborLand(i - 1, j);
        findNeighborLand(i, j - 1);
        findNeighborLand(i + 1, j);
        findNeighborLand(i, j + 1);
    };

    for (size_t i = 0; i < grid.size(); ++i)
    {
        for (size_t j = 0; j < grid[i].size(); ++j)
        {
            if (grid[i][j] == LAND)
            {
                ++count;
                findNeighborLand(i, j);
            }
        }
    }

    return count;
}

vector<string> Solution::letterCombinations(string digits)
{
    const int digitsSize = digits.size();

    if (digitsSize == 0) return {};

    static const std::vector<std::vector<char>> numberChars =
    {
        {{'a','b','c'}},    // 2
        {{'d','e','f'}},    // 3
        {{'g','h','i'}},    // 4
        {{'j','k','l'}},    // 5
        {{'m','n','o'}},    // 6
        {{'p','q','r','s'}},// 7
        {{'t','u','v'}},    // 8
        {{'w','x','y','z'}},// 9
    };

    std::vector<size_t> indices(digitsSize, 0);

    vector<string> result;

    const int firstDigit = digits[0] - '0';

    while (indices[0] < numberChars[firstDigit - 2].size())
    {
        string comb;
        for (int i = 0; i < digitsSize; ++i)
        {
            const int digit = digits[i] - '0';
            comb.push_back(numberChars[digit - 2][indices[i]]);
        }

        result.emplace_back(comb);

        for (size_t i = indices.size() - 1; i >= 0; --i)
        {
            ++indices[i];

            const int digit = digits[i] - '0';
            if (indices[i] < numberChars[digit - 2].size())
            {
                break;
            }

            if (i != 0)
            {
                indices[i] = 0;
            }
        }
    }

    return result;
}

vector<string> Solution::letterCombinationsVariant(string digits)
{
    vector<string> result;
    unordered_map<int, vector<string>> mp = 
    {
        {2, {"a","b","c"}},
        {3, {"d","e","f"}},
        {4, {"g","h","i"}},
        {5, {"j","k","l"}},
        {6, {"m","n","o"}},
        {7, {"p","q","r","s"}},
        {8, {"t","u","v"}},
        {9, {"w","x","y","z"}}
    };

    for (char c : digits) 
    {
        const int i = c - '0';

        if (result.empty()) result = mp[i];
        else 
        {
            vector<string> aux;
            for (const string& str : mp[i]) 
            {
                for (const string& r : result)
                {
                    aux.push_back(r + str);
                }
            }

            result = aux;
        }
    }
    return result;
}

int Solution::maxSubArray(vector<int>& nums)
{
    // Using Kadane's Algorithm

    int result = 0;
    int maxSum = INT_MIN;

    for (size_t i = 0; i < nums.size(); ++i)
    {
        maxSum = max(nums[i], maxSum + nums[i]);
        result = max(result, maxSum);
    }

    return result;
}

int Solution::maxSubarraySumCircular(vector<int>& nums)
{
    // Using Kadane's Algorithm

    int totalSum = 0;
    int minSum = nums[0], maxSum = nums[0];
    int currentMaxSum = 0, currentMinSum = 0;

    for (size_t i = 0; i < nums.size(); i++)
    {
        currentMaxSum = max(currentMaxSum + nums[i], nums[i]);
        maxSum = max(maxSum, currentMaxSum);

        currentMinSum = min(currentMinSum + nums[i], nums[i]);
        minSum = min(minSum, currentMinSum);

        totalSum += nums[i];
    }

    if (minSum == totalSum)
    {
        return maxSum;
    }

    return max(maxSum, totalSum - minSum);
}

int Solution::findKthLargest(vector<int>& nums, int k)
{
    nth_element(nums.begin(), nums.begin() + k - 1, nums.end(), greater<int>());
    return nums[k - 1];
}

bool Solution::canJump(vector<int>& nums)
{
    // Using Greedy algorithm

    int indexToAchieve = nums.size() - 1;

    for (int i = indexToAchieve - 1; i >= 0; --i)
    {
        if (indexToAchieve <= nums[i] + i)
        {
            indexToAchieve = i;
        }
    }

    return indexToAchieve == 0;
}
