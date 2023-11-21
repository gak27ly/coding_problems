/*
1. TwoSum
利用HashMap
*/
public int[] twoSum(int[] nums, int target) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++){
        if (map.get(nums[i]) != null){
            int[] res = {map.get(nums[i]), i};
            return res;
        }
        map.put(target - nums[i], i);
    }
    int[] res = {};
    return res;
}



public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode l3 = new ListNode(0);
    ListNode dummy = l3;
    int carry = 0;
        
    while (l1 != null || l2 != null || carry != 0){
        int total = carry;
        /*
        total += l1 != null? l1.val : 0;
        total += l2 != null? l2.val : 0;
        l1 = l1 != null? l1.next : null;
        l2 = l2 != null? l2.next : null;
        */
        if (l1 != null){
            total += l1.val;
            l1 = l1.next;
        }
        if (l2 != null){
            total += l2.val;
            l2 = l2.next;
        }

        l3.next = new ListNode(total % 10);
        l3 = l3.next;
        carry = total / 10;
    }
    return dummy.next;  
}

/*
3. Longest Substring Without Repeating Characters
解法：利用hashSet来遍历string
*/
public int lengthOfLongestSubstring(String s) {
    HashSet map = new HashSet();
    int l = 0;
    int maxLength = 0;
    for (int r = 0; r < s.length(); r++){
        while (map.contains(s.charAt(r))){
            map.remove(s.charAt(l));
            l++;
        }
        map.add(s.charAt(r));
        maxLength = Math.max(maxLength, r - l + 1);
    }
    return maxLength;
}

/* 
5. Longest Palindromic Substring
注意点： 当maxLen是偶数时，注意start的取值为 i - (maxLen - 1) / 2
*/
public String longestPalindrome(String s) {
    int len = s.length(), start = 0, end = 0;
    int maxLen = 0;
    
    for (int i = 0; i < len; i++){
        int currLen = Math.max(palindromLength(s, i, i), palindromLength(s, i, i + 1));
        if (currLen > maxLen){
            maxLen = currLen;
            start = i - (maxLen - 1) / 2;
        }
    }
    return s.substring(start, start + maxLen);
}
   
private int palindromLength(String s, int left, int right){        
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
    }
    return right - left - 1;
}
/*
6. ZigZag Conversion
注意点： 倒序时候从numRows - 2开始.
*/

public String convert(String s, int numRows) {
    StringBuilder[] sb = new StringBuilder[numRows];
    
    for (int i = 0; i < numRows; i++) sb[i] = new StringBuilder();
    
    int len = s.length();
    int index = 0;
    while (index < len){
        for (int i = 0; i < numRows && index < len; i++){
            sb[i].append(s.charAt(index++));                
        }
        
        for (int i = numRows - 2; i > 0 && index < len; i--){
            sb[i].append(s.charAt(index++));
        }
    }
    for (int i = 1; i < numRows; i++){
        sb[0].append(sb[i]);
    }
    return sb[0].toString();
}


/*
8. String to Intege
注意点： java中需要判断integer边界
*/
public int myAtoi(String s) {
    int res = 0;
    int i = 0;
    int sign = 1;
    
    while (i < s.length() && s.charAt(i) == ' ') 
        i++;
    
    if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')){
        sign = s.charAt(i++) == '+' ? 1 : -1;
    }
    
    while (i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9'){
        int reminder = s.charAt(i++) - '0';
        int newRes = res * 10 + reminder;
        if (newRes / 10 != res){
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        res = newRes;
    }
    return res * sign;
}


/*
11. Container With Most Water
*/

public int maxArea(int[] height) {
    int left = 0, right = height.length - 1;
    int maxVolumn = 0;
    
    while (left < right){
        maxVolumn = Math.max(maxVolumn, (right - left) * Math.min(height[left], height[right]));
        if (height[left] < height[right])
            left++;
        else
            right--;
    }
    return maxVolumn;
}

/*
14. Longest Common Prefix
*/
public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) return "";
    String res = "";
    int index = 0;
    
    for (char c: strs[0].toCharArray()){
        for (int i = 1; i < strs.length; i++){
            if (index >= strs[i].length() || c != strs[i].charAt(index)){
                return res;
            }
        }
        res += c;
        index++;
    }
    return res;
}

/*
15. 3Sum
注意点： 需要在index上去重，还需要在left上去重
*/
public List<List<Integer>> threeSum(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList();
    
    for (int i = 0; i < nums.length - 2; i++){
        if (i != 0 && nums[i] == nums[i - 1]) 
            continue;
        twoSum(nums, i, nums.length - 1, res);
    }
    return res;
}

private void twoSum(int[] nums, int index, int right, List<List<Integer>> res){
    int left = index + 1, target = -nums[index];
    while (left < right){
        if (nums[left] + nums[right] < target){
            left++;
        } else if(nums[left] + nums[right] > target){
            right--;
        } else{
            List<Integer> triple = new ArrayList();
            triple.add(nums[index]);
            triple.add(nums[left]);
            triple.add(nums[right]);
            res.add(triple);
            left++;
            right--;
            while (left < right && nums[left] == nums[left - 1]){
                left++;
            }
        }
    }
}

/*
16. 3Sum Closest
注意点： 通过确定一个点再来找三个数字和与target的最小差值
*/

public int threeSumClosest(int[] nums, int target) {
    if (nums == null || nums.length == 0) return target;
    Arrays.sort(nums);
    int diff = Integer.MAX_VALUE;
    for (int i = 0; i < nums.length - 2; i++){
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        
        int left = i + 1, right = nums.length - 1;
        while (left < right){
            int total = nums[i] + nums[left] + nums[right];
            if (total == target)
                return target;
            if (Math.abs(total - target) < Math.abs(diff))
                diff = total - target;
            if (total < target)
                left++;
            else
                right--;
        }
    }
    return target + diff;
}

/*
17. Letter Combinations of a Phone Number

*/
public List<String> letterCombinations(String digits) {
    HashMap<Character, String> mapping = new HashMap();
    mapping.put('2', "abc");
    mapping.put('3', "def");
    mapping.put('4', "ghi");
    mapping.put('5', "jkl");
    mapping.put('6', "mno");
    mapping.put('7', "pqrs");
    mapping.put('8', "tuv");
    mapping.put('9', "wxyz");
    
    List<String> res = new ArrayList();
    if (digits == null || digits.length() == 0)
        return res;
    
    dfs("", 0, digits, res, mapping);
    return res;
}

private void dfs(String curr, int index, String digits, List res, HashMap<Character, String> mapping){
    if (index == digits.length()){
        res.add(curr);
    } else{
        char c = digits.charAt(index);
        for (char ch : mapping.get(c).toCharArray()){
            dfs(curr + ch, index + 1, digits, res, mapping);
        }
    }
}

/*
19. Remove Nth Node From End of List
*/
public ListNode removeNthFromEnd(ListNode head, int n) {
    if (head == null) return null;
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode slow = dummy;
    ListNode fast = dummy;
    
    for (int i = 0; i < n; i++){
        fast = fast.next;
    }
    
    while (fast != null && fast.next != null){
        slow = slow.next;
        fast = fast.next;
    }

    slow.next = slow.next.next;
    return dummy.next;      
}

/*
20. Valid Parentheses
再做
*/
public boolean isValid(String s) {
    if (s == null || s.length() <= 1) return false;
    if (s.length() % 2 != 0) return false;
    
    HashMap<Character, Character> mapping = new HashMap<Character, Character>();
    Stack<Character> stack = new Stack<Character>();
    mapping.put(')' , '(');
    mapping.put(']' , '[');
    mapping.put('}' , '{');
    
    for (int i = 0; i < s.length(); i++){
        char c = s.charAt(i);
        if (!mapping.containsKey(c))
            stack.push(c);
        
        else{

            if (stack.isEmpty() || mapping.get(c) != stack.pop()) return false;
        }
    }
    return stack.isEmpty();
}

/*
22. Generate Parentheses
*/

public List<String> generateParenthesis(int n) {
    ArrayList<String> res = new ArrayList();
    dfs(0, 0, n, "", res);
    return res;    
}

private void dfs(int left, int right, int n, String combination, ArrayList res){
    if (left > n || right > n)
        return;
    
    if (right > left){
        return;
    }
    
    if (right == n && left == n){
        res.add(combination);
        return;
    }
    
    dfs(left + 1, right, n, combination + "(", res);
    dfs(left, right + 1, n, combination + ")", res);
}

/*
23. Merge k Sorted Lists
解法： 利用divide&conquer
也可以从头开始，两两相连，连接k次
再做
*/
public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
        return null;
    return mergeListRange(0, lists.length - 1, lists);
}

private ListNode mergeListRange(int start, int end, ListNode[] lists) {
    if (start == end){
        return lists[start];
    }
    int mid = start + (end - start) / 2;

    ListNode left = mergeListRange(start, mid, lists);
    ListNode right = mergeListRange(mid + 1, end, lists);
    return mergeLists(left, right);
}

private ListNode mergeLists(ListNode left, ListNode right) {
    ListNode dummy = new ListNode(0);
    ListNode head = dummy;
    while (left != null && right != null) {
        if (left.val < right.val){
            head.next = left;
            left = left.next;
        } else {
            head.next = right;
            right = right.next;
        }
        head = head.next;
    }
    if (left != null)
        head.next = left;
    if (right != null)
        head.next = right;

    return dummy.next;
}

/*
24. Swap Nodes in Pairs
*/
public ListNode swapPairs(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode dummy = new ListNode(0);
    ListNode pre = dummy;
    ListNode l1 = head;
    ListNode l2 = head.next;

    while (l1 != null && l2 != null){
        pre.next = l2;
        l1.next = l2.next;
        l2.next = l1;
        pre = l1;
        l1 = l1.next;
        if (l1 != null)
            l2 = l1.next;
    }
    return dummy.next;
}

/*
26. Remove Duplicates from Sorted Array
解法：先定下前点表示已经排好序的位置i
从后面找到一个大于nums[i]的数字放到i+1的位置.
*/

public int removeDuplicates(int[] nums) {
            
    if (nums == null ||  nums.length == 0) return 0;
    
    int i = 0;
    
    for (int j = 1; j < nums.length; j++){
        if (nums[i] < nums[j]){
            i++;
            nums[i] = nums[j];
        }
    }
    return i + 1;
}

/*
27. Remove Element
解法： i位置为已经排好序的位置，把j位置的非val值传给i位置
只有当i位置收到新的值时才会移动到下一个位置
*/

public int removeElement(int[] nums, int val) {
    if (nums == null) return 0;
    int i = 0;

    for (int j = 0; j < nums.length; j++){
        if (nums[j] != val){
            nums[i] = nums[j];
            i++;
        }
    }
    return i;
}

/*
28. Implement strStr()
*/

public int strStr(String haystack, String needle) {
    if (haystack == null || needle == null) return -1;

    int nLen = needle.length();
    for (int i = 0; i < haystack.length() - nLen + 1; i++){
        int j = 0;
        for (j = 0; j < nLen; j++){
            if (needle.charAt(j) != haystack.charAt(i + j)){
                break;
            }
        }
        if (j == nLen) 
            return i;
    }
    return -1;
}

/*
31. Next Permutation
*/

public void nextPermutation(int[] nums) {
    int i = nums.length - 1;
    while (i >= 1 && nums[i] <= nums[i - 1])
        i--;
    if (i == 0) {
        reverse(nums, 0, nums.length - 1);
        return; 
    }

    for (int j = nums.length - 1; j > i - 1; j--){
        if (nums[j] > nums[i - 1]){
            int temp = nums[i - 1];
            nums[i - 1] = nums[j];
            nums[j] = temp;
            break;
        }
    }
    reverse(nums, i, nums.length - 1);
}

private void reverse(int[] nums, int start, int end){
    while (start < end){
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}

/*
33. Search in Rotated Sorted Array
*/
public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0)
        return -1;
    int start = 0;
    int end = nums.length - 1;

    while (start + 1 < end){
        int mid = start + (end - start) / 2;
        if (nums[mid] > nums[end]){
            if (nums[start] <= target && target <= nums[mid])
                end = mid;
            else
                start = mid;
        }
        else{
            if (nums[mid] <= target && target <= nums[end])
                start = mid;
            else
                end = mid;
        }
    }
    if (nums[start] == target) 
        return start;
    if (nums[end] == target)
        return end;
    return -1;
}



/*
39. Combination Sum
*/

public List<List<Integer>> combinationSum(int[] candidates, int target) {
    if (candidates == null) return null;
    Arrays.sort(candidates);
    List<List<Integer>> res = new ArrayList<>();
    dfs(candidates, 0, target, new ArrayList<>(), res);
    return res;
}

private void dfs(int[] nums, 
                 int index, 
                 int target, 
                 List<Integer>combination, 
                 List<List<Integer>> res){
    if (target < 0)
        return;

    if (target == 0){
        res.add(new ArrayList<>(combination));
        return;
    }

    for (int i = index; i < nums.length; i++){
        combination.add(nums[i]);
        dfs(nums, i, target - nums[i], combination, res);
        combination.remove(combination.size() - 1);
    }    
}

/*
40. Combination Sum II
*/

public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(candidates);
    dfs(candidates, 0, target, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int[] nums,
                 int index,
                 int target, 
                 List<Integer> combination, 
                 List<List<Integer>>res){
    if (target == 0){
        res.add(new ArrayList<>(combination));
        return;
    }

    if (target < 0)
        return;
    for (int i = index; i< nums.length; i++){
        if (i != index && nums[i] == nums[i - 1])
            continue;
        combination.add(nums[i]);
        dfs(nums, i + 1, target - nums[i], combination, res);
        combination.remove(combination.size() - 1);
    }
}

/*
42. Trapping Rain Water
two pointer低的挡板决定当前位置的水量上限.
*/
public int trap(int[] height) {
    if (height == null || height.length == 0)
        return 0;
    int lMax = height[0];
    int rMax = height[height.length - 1];
    int res = 0;
    int l = 0;
    int r = height.length - 1;

    while (l < r){
        if (lMax < rMax){
            res += lMax - height[l];
            l += 1;
            lMax = Math.max(lMax, height[l]);
        } else{
            res += rMax - height[r];
            r -= 1;
            rMax = Math.max(rMax, height[r]);
        }
    }
    return res;
}

/*
43. Multiply Strings
解法： 利用相乘的原理先将数字两辆相乘放入数组，再处理进位
*/
public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0"))
        return "0";
    int[] res = new int[num1.length() + num2.length()];
    for (int i = 0; i < num2.length(); i++){
        int a = num2.charAt(i) - '0';
        for (int j = 0; j < num1.length(); j++){
            int b = num1.charAt(j) - '0';
            res[i + j + 1] += a * b;
        }
    }

    for (int i = res.length - 1; i > 0; i--){
        res[i - 1] += res[i] / 10;
        res[i] = res[i] % 10;
    }

    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < res.length; i++){
        if (res[i] == 0 && sb.length() == 0)
            continue;
        sb.append(res[i]);
    }
    return sb.toString();
}

/*
46. Permutations
用一个boolean[]来记录被visited过的点即可，比arraylist速度快
*/
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();

    dfs(nums, new boolean[nums.length], new ArrayList(), res);
    return res;
}

private void dfs(int[] nums, 
                 boolean[] visited, 
                 List<Integer> permutation, 
                 List<List<Integer>> res) {

    if (permutation.size() == nums.length){
        res.add(new ArrayList<Integer>(permutation));
        return;
    }

    for (int i = 0; i < nums.length; i++){
        if (visited[i])
            continue;

        visited[i] = true;
        permutation.add(nums[i]);
        dfs(nums, visited, permutation, res);
        visited[i] = false;
        permutation.remove(permutation.size() - 1);
    }
}

/*
47. Permutations II
*/
public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> res = new ArrayList();
    Arrays.sort(nums);
    boolean[] visited = new boolean[nums.length];
    dfs(nums, visited, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int[] nums,
                boolean[] visited,
                List<Integer> permutation,
                List<List<Integer>> res){

    if (permutation.size() == nums.length){
        res.add(new ArrayList<Integer>(permutation));
        return;
    }

    for (int i = 0; i < nums.length; i++){
        if (visited[i])
            continue;
        //判断相同的前一个数字是否已被取用，若没有则不能取用当前数字
        if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])
            continue;
        visited[i] = true;
        permutation.add(nums[i]);
        dfs(nums, visited, permutation, res);
        visited[i] = false;
        permutation.remove(permutation.size() - 1);
    }
}

/*
48. Rotate Image
简单
*/

public void rotate(int[][] matrix) {

    for (int i = 0; i < matrix.length; i++){
        for (int j = i + 1; j < matrix[0].length; j++){
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp; 
        }
    }

    int start = 0;
    int end = matrix[0].length - 1;

    for (int i = 0; i < matrix.length; i++){
        reverse(matrix[i], start, end);
    }
}

private void reverse(int[] row, int start, int end){
    while (start < end){
        int temp = row[start];
        row[start] = row[end];
        row[end] = temp;
        start += 1;
        end -= 1;
    }
}

/*
49. Group Anagrams

*/

public List<List<String>> groupAnagrams(String[] strs) {
    if (strs == null || strs.length == 0)
        return new ArrayList<List<String>>();
    HashMap<String, List<String>> map = new HashMap<String, List<String>>();

    for (int i = 0; i < strs.length; i++){
        char[] chs = strs[i].toCharArray();
        Arrays.sort(chs);
        String key = String.valueOf(chs);
        if (!map.containsKey(key)){
            map.put(key, new ArrayList<String>());
        } 
        map.get(key).add(strs[i]);
    }
    return new ArrayList<List<String>>(map.values());
}


/*
50. Pow(x, n)
解法： 二分法
*/

public double myPow(double x, int n) {
    if (x == 0 || n == 0) 
        return 1;

    long m = n;
    if (m < 0){
        x = 1 / x;
        m = -m;
    }

    double res = 1;
    double num = x;
    while (m > 0){
        if (m % 2 == 1){
            res *= num;
            m -= 1;
        }

        num = num * num;
        m = m / 2;
    }

    return res;
}
/*
50. Pow(x, n)
解法： 二分法
*/

public double myPow(double x, int n) {
    if (x == 0 || n == 0) 
        return 1;

    long m = n;
    if (m < 0){
        x = 1 / x;
        m = -m;
    }

    double res = 1;
    double num = x;
    while (m > 0){
        if (m % 2 == 1){
            res *= num;
            m -= 1;
        }

        num = num * num;
        m = m / 2;
    }
    return res;
}

/*
51. N-Queens
*/

public List<List<String>> solveNQueens(int n) {
    if (n <= 0) return new ArrayList<List<String>>();
    List<List<String>> res = new ArrayList();
    dfs(n, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int n, List<Integer> cols, List<List<String>> res) {
    if (cols.size() == n) {
        res.add(draw(cols));
        return;
    }
    for (int row = 0; row < n; row++) {
        if (!isValid(cols, row))
            continue;
        cols.add(row);
        dfs(n, cols, res);
        cols.remove(cols.size() - 1);
    }
}

private boolean isValid(List<Integer> cols, int newRow) {

    for (int i = 0; i < cols.size(); i++){
        if (cols.get(i) == newRow)
            return false;
        if (Math.abs(cols.get(i) - newRow) == Math.abs(i - cols.size()))
            return false;
    }
    return true;
}

private List<String> draw(List<Integer> cols){
    List<String> res = new ArrayList<String>();
    for (int i = 0; i < cols.size(); i++){
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < cols.size(); j++){
            sb.append(cols.get(i) == j ? "Q" : ".");
        }
        res.add(sb.toString());
    }
    return res;
}

/*
54. Spiral Matrix
简单题，注意loop结束条件
*/
public List<Integer> spiralOrder(int[][] matrix) {
    if (matrix == null || matrix.length == 0)
        return new ArrayList<Integer>();
    int left = 0;
    int right = matrix[0].length - 1;
    int top = 0;
    int bot = matrix.length - 1;
    int dir = 0;
    List<Integer> res = new ArrayList<Integer>();

    while (left <= right && top <= bot){
        if (dir == 0){
            for (int i = left; i < right + 1; i++) 
                res.add(matrix[top][i]);
            top++;
        } else if (dir == 1) {
            for (int i = top; i < bot + 1; i++) 
                res.add(matrix[i][right]);
            right--;

        } else if (dir == 2) {
            for (int i = right; i > left - 1; i--)
                res.add(matrix[bot][i]);
            bot--;
        } else if (dir == 3) {
            for (int i = bot; i > top - 1; i--)
                res.add(matrix[i][left]);
            left++;
        }
        dir = (dir + 1) % 4;
    }
    return res;
}

/*
56. Merge Intervals
解法：对起始点进行排序，然后利用stack或者linkedlist的getLast()
来获取之前已加入的interval的结尾范围，再判断是否合并.
*/
public int[][] merge(int[][] intervals) {
    if (intervals == null || intervals.length == 0)
        return new int[0][0];
    
    LinkedList<int[]> res = new LinkedList<>();
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    
    for (int[] interval : intervals) {
        if (res.size() == 0 || interval[0] > res.getLast()[1])
            res.add(interval);
        else
            res.getLast()[1] = Math.max(res.getLast()[1], interval[1]);
    }
    
    return res.toArray(new int[res.size()][2]);        
}

/*
57. Insert Interval
*/
public int[][] insert(int[][] intervals, int[] newInterval) {
    LinkedList<int[]> res = new LinkedList<int[]>();
    if (intervals == null || intervals.length == 0) {
        res.add(newInterval);
        return res.toArray(new int[res.size()][2]);

    }
    int index = 0;
    int n = intervals.length;
    int start = newInterval[0];
    int end = newInterval[1];

    while (index < n && intervals[index][1] < newInterval[0]) {
        res.add(intervals[index]);
        index++;
    }
    if (index == n) {
        res.add(newInterval);
        return res.toArray(new int[res.size()][2]);
    } 
    start = Math.min(intervals[index][0], newInterval[0]);

    while (index < n && intervals[index][0] <= newInterval[1]) {
        end = Math.max(intervals[index][1], end);
        index++;
    }

    res.add(new int[] {start, end});

    while (index < n) {
        res.add(intervals[index++]);
    }

    return res.toArray(new int[res.size()][2]);
}


/*
77. Combinations
*/

public List<List<Integer>> combine(int n, int k) {

    List<List<Integer>> res = new ArrayList<>();

    dfs(n, k, 1, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int n, int k, int startIndex, List<Integer> combination, List<List<Integer>> res) {

    if (combination.size() == k) {
        res.add(new ArrayList<Integer>(combination));
        return;
    }

    for (int i = startIndex; i <= n; i++) {
        combination.add(i);
        dfs(n, k, i + 1, combination, res);
        combination.remove(combination.size() - 1);
    }
}
    return res.toArray(new int[res.size()][2]);
}

/*
66. Plus One
*/

public int[] plusOne(int[] digits) {
    if (digits == null || digits.length == 0)
        return new int[] {};

    int n = digits.length;

    for (int i = n - 1; i >= 0; i--) {
        if (digits[i] != 9) {
            digits[i]++;
            return digits;
        }
        digits[i] = 0;
    }

    int[] newInt = new int[n + 1];
    newInt[0] = 1;
    return newInt;
}

/*
69. Sqrt(x)
*/
public int mySqrt(int x) {
    if (x < 1)
        return 0;

    int start = 1;
    int end = (int)Math.sqrt(Integer.MAX_VALUE);

    while (start + 1 < end) {
        int mid = start + (end - start) / 2;
        if (mid * mid > x) {
            end = mid;
        } else if (mid * mid < x) {
            start = mid;
        } else {
            return mid;
        }
    }

    if (end * end <= x)
        return end;
    return start;
}

/*
75. Sort Colors
*/
public void sortColors(int[] nums) {
    if (nums == null || nums.length == 0)
        return;

    int p1 = 0;
    int p2 = 0;
    int p3 = nums.length - 1;

    while (p2 <= p3) {
        if (nums[p2] == 0) {
            swap(nums, p1, p2);
            p1++;
            p2++;
        } else if (nums[p2] == 2) {
            swap(nums, p2, p3);
            p3--;
        } else {
            p2++;
        }
    }
}

private void swap(int[] nums, int a, int b) {
    int temp = nums[a];
    nums[a] = nums[b];
    nums[b] = temp;
}

/*
78. Subsets
*/
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    if (nums == null || nums.length == 0)
        return res;
    
    dfs(nums, 0, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int[] nums, int startIndex, List<Integer> subset, List<List<Integer>> res) {
    
    res.add(new ArrayList<Integer>(subset));
    
    for (int i = startIndex; i < nums.length; i++) {
        subset.add(nums[i]);
        dfs(nums, i + 1, subset, res);
        subset.remove(subset.size() - 1);
    }
}

/*
<<<<<<< HEAD
100. Same Tree
简化前面的判断
*/
public boolean isSameTree(TreeNode p, TreeNode q) {
    
    if (p == null && q == null) {
        return true;
    } else if (p == null || q == null) {
        return false;
    }
    
    boolean sameLeft = isSameTree(p.left, q.left);
    boolean sameRight = isSameTree(p.right, q.right);

    return p.val == q.val && sameLeft && sameRight;
}

/*
101. Symmetric Tree
需要一个helper function来帮助判断两个点
*/

public boolean isSymmetric(TreeNode root) {
    if (root  == null) return true;

    return checkSymmetric(root.left, root.right);      
}

private boolean checkSymmetric(TreeNode leftNode, TreeNode rightNode) {
    if (leftNode == null && rightNode == null) {
        return true;
    }
    if (leftNode == null || rightNode == null) {
        return false;
    }

    return leftNode.val == rightNode.val && checkSymmetric(leftNode.left, rightNode.right)
        &&  checkSymmetric(leftNode.right, rightNode.left);


}

/*
102. Binary Tree Level Order Traversal
要把levelList加入res，res需要先initialize
*/
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList();
    if (root == null) return res;
    Queue<TreeNode> nodeQueue = new LinkedList();
    nodeQueue.offer(root);

    while(!nodeQueue.isEmpty()) {
        int levelSize = nodeQueue.size();
        List<Integer> levelList = new ArrayList();
        for (int i = 0; i < levelSize; i++) {
            TreeNode node = nodeQueue.poll();
            if (node.left != null) {
                nodeQueue.offer(node.left);
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
            }
            levelList.add(node.val);
        }
        res.add(levelList);
    }
    return res;        
}

/*
104. Maximum Depth of Binary Tree
*/
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    if (root.left == null && root.right == null) return 1;
    
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
}

/*
111. Minimum Depth of Binary Tree
当一边没有点的时候要看旁边的subtree，直到找到leaf
*/
public int minDepth(TreeNode root) {
    if (root == null) return 0;
    if (root.left == null && root.right == null) return 1;
    if (root.left == null) return minDepth(root.right) + 1;
    if (root.right == null) return minDepth(root.left) + 1;

    return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
}
/*
112. Path Sum
如上题，需要保证有一条root到leaf的path
*/

public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) return false;
    if (root.left == null && root.right == null) return root.val == targetSum;
    if (root.left == null) return hasPathSum(root.right, targetSum - root.val);
    if (root.right == null) return hasPathSum(root.left, targetSum - root.val);
    
    return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
}

/*
116. Populating Next Right Pointers in Each Node
需要理解方法二
*/

public Node connect(Node root) {

    if (root == null) return root;

    Queue<Node> nodeQueue = new LinkedList();

    nodeQueue.offer(root);

    while (!nodeQueue.isEmpty()) {
        int levelSize = nodeQueue.size();
        for (int i = 0; i < levelSize; i++) {
            Node node = nodeQueue.poll();
            node.next = nodeQueue.peek();
            if (i == levelSize - 1) {
                node.next = null;
            }
            if (node.left != null) {
                nodeQueue.offer(node.left);
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
            }
        }
    }
    
    return root;
}

public Node connect(Node root) {
    Node answ = root;
    Node tmpLeft;
    while (root != null) {
        tmpLeft = root.left;
        while (root != null && root.left != null) {
                root.left.next = root.right;
            if (root.next != null)
                root.right.next = root.next.left;
            root = root.next;
        }
        root = tmpLeft;
    }
    return answ;
}


/*
133. Clone graph
nodeQueue.isEmpty 写成了 nodeQueue == null
mapping.put(popedNode, new Node(popedNode.val)) 写到了for loop里面
*/
public Node cloneGraph(Node node) {
    if (node == null) {
        return node;
    }

    Map<Node, Node> mapping = new HashMap();
    Queue<Node> nodeQueue = new LinkedList();

    nodeQueue.offer(node);

    while (!nodeQueue.isEmpty()) {
        Node popedNode = nodeQueue.poll();
        mapping.put(popedNode, new Node(popedNode.val));
        for (Node neighbor : popedNode.neighbors) {
            if (mapping.get(neighbor) == null) {
                nodeQueue.offer(neighbor);
            }
        }
    }

    for (Node oldNode : mapping.keySet()) {
        for (Node neighbor : oldNode.neighbors) {
            mapping.get(oldNode).neighbors.add(mapping.get(neighbor));
        }
    }

    return mapping.get(node);
}


/*
199. Binary Tree Right Side View
*/
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList();
    if (root == null) return res;

    Queue<TreeNode> q = new LinkedList();

    q.offer(root);
    while (!q.isEmpty()) {
        int levelSize = q.size();
        while (levelSize > 0) {
            levelSize--;
            TreeNode node = q.poll();
            if (levelSize == 0) {
                res.add(node.val);
            }
            if (node.left != null) {
                q.offer(node.left);
            }
            if (node.right != null) {
                q.offer(node.right);
            }
        }
    
    }
    return res;

}

/*
200. Number of Islands
注意需要立刻改变押入queue的点的值来保证不重复读取
若不让改变值，则需要建立一个对应的boolean[][]来帮助判断是否visited
*/
public int numIslands(char[][] grid) {
    int numOfIslands = 0;
    if (grid == null || grid[0].length == 0) {
        return numOfIslands;
    }

    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == '1') {
                bfs(i, j, grid);
                numOfIslands++;
            }
        }
    }
    return numOfIslands;
}

private void bfs(int i, int j, char[][] grid) {
    Queue<Integer> rowQueue = new LinkedList();
    Queue<Integer> colQueue = new LinkedList();
    int[] rowDirection = {0, 0, 1, -1};
    int[] colDirection = {1, -1, 0, 0};

    rowQueue.offer(i);
    colQueue.offer(j);
    grid[i][j] = '0';

    while (!rowQueue.isEmpty() && !colQueue.isEmpty()) {
        int row = rowQueue.poll();
        int col = colQueue.poll();

        for (int k = 0; k < rowDirection.length; k++) {
            int newRow = row + rowDirection[k];
            int newCol = col + colDirection[k];
            if (isValid(newRow, newCol, grid)) {
                rowQueue.offer(newRow);
                colQueue.offer(newCol);
                grid[newRow][newCol] = '0';
            }
        }
    }

}

private boolean isValid(int row, int col, char[][] grid) {
    if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length) {
        return false;
    }
    return grid[row][col] == '1';
}


/*
207. Course Schedule
需注意Map读取时候的检查mapping.containsKey(classNum)
*/

public boolean canFinish(int numCourses, int[][] prerequisites) {
    
    Map<Integer, List<Integer>> mapping = new HashMap();
    int[] inDegree = new int[numCourses];


    for (int[] prerequisite : prerequisites) {
        if(mapping.containsKey(prerequisite[1])) {
            mapping.get(prerequisite[1]).add(prerequisite[0]);
        } else {
            mapping.put(prerequisite[1], new ArrayList(List.of(prerequisite[0])));
        }
        inDegree[prerequisite[0]]++;
    }

    bfs(mapping, inDegree);

    for (int i = 0; i < inDegree.length; i++) {
        if (inDegree[i] != 0) {
            return false;
        }
    }
    return true;
}

private void bfs(Map<Integer, List<Integer>> mapping, int[] inDegree) {
    Queue<Integer> q = new LinkedList();
    for (int i = 0; i < inDegree.length; i++) {
        if (inDegree[i] == 0) {
            q.offer(i);
        }
    }

    while (!q.isEmpty()) {
        int classNum = q.poll();
        if (!mapping.containsKey(classNum)) {
            continue;
        }
        for (int nextClass : mapping.get(classNum)) {
            inDegree[nextClass]--;
            if (inDegree[nextClass] == 0) {
                q.offer(nextClass);
            } 
        }
    }
=======
79. Word Search
*/
public boolean exist(char[][] board, String word) {
    if (word == null || word.length() == 0)
        return true;
    if (board == null || board.length == 0)
        return false;

    int m = board.length;
    int n = board[0].length;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == word.charAt(0)) {
                if (search(board, i, j, word, 0) == true)
                    return true;
            }
        }
    }
    return false;
}

private boolean search(char[][]board, int i, int j, String word, int idx) {
    if (idx == word.length())
        return true;
    if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] != word.charAt(idx))
        return false;

    board[i][j] = '#';
    boolean found = search(board, i + 1, j, word, idx + 1) || 
                    search(board, i - 1, j, word, idx + 1) ||
                    search(board, i, j + 1, word, idx + 1) ||
                    search(board, i, j - 1, word, idx + 1);

    board[i][j] = word.charAt(idx);
    return found;
>>>>>>> b110260... coding problems
}
