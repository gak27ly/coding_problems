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