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