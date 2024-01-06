//1. If there is a duplicate
public boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int i = 0; i < nums.length; i++) {
        if (set.contains(nums[i])) {
            return true;
        }
        set.add(nums[i]);
    }
    return false;
}

//2.  if two strings have same characters
public boolean isAnagram(String s, String t) {
    if (s == null || t == null) return false;
    if (s.length() != t.length()) return false;

    int[] charCount = new int[26];

    for (int i = 0; i < s.length(); i++) {
        charCount[s.charAt(i) - 'a']++;
        charCount[t.charAt(i) - 'a']--;
    }

    for (int num : charCount) {
        if (num != 0) return false;
    }
    return true;
}

//3 two sum
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> mapping = new HashMap<>();

    for (int index = 0; index < nums.length; index++) {
        int diff = target - nums[index];
        if (mapping.containsKey(diff)) {
            return new int[]{mapping.get(diff), index};
        }
        mapping.put(nums[index], index);
    }
    return new int[] {};
}

//4 Group Anagrams
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap();

    for (String str : strs) {
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        String sortedWord = new String(chars);
        map.putIfAbsent(sortedWord, new ArrayList());
        map.get(sortedWord).add(str);
    }
    return new ArrayList(map.values());
}

//5. Top K elements
/**
 * Time Complexity: O(nlog(k))
 * Space Complexity: O(n)
 */

public int[] topKFrequent(int[] nums, int k) {
    int[] arr = new int[k];
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int num : nums) map.put(num, map.getOrDefault(num, 0) + 1);
    PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>(
            (a, b) ->
        a.getValue() - b.getValue()
    );
    for (Map.Entry<Integer, Integer> it : map.entrySet()) {
        pq.add(it);
        if (pq.size() > k) pq.poll();
    }
    int i = k;
    while (!pq.isEmpty()) {
        arr[--i] = pq.poll().getKey();
    }
    return arr;
}

 /**
 * Time Complexity: O(n)
 * Space Complexity: O(n)
 * get map of occurance, and put the most frequesnt number into a countArray 
 * so the index of the countArray holds all numbers has index number of occurance
 */
public int[] topKFrequent(int[] nums, int k) {
    if (nums == null || nums.length == 0) {
        return new int[]{};
    }
    Map<Integer, Integer> counts = new HashMap<>();

    for (int num : nums) {
        counts.putIfAbsent(num, 0);
        counts.put(num, counts.get(num) + 1);
    }

    List<Integer> countArray[] = new ArrayList[nums.length + 1];                

   for(int key : counts.keySet()) {
        int value = counts.get(key);
        if (countArray[value] == null) {
            countArray[value] = new ArrayList<Integer>();
        }
        countArray[value].add(key);
    }

    int[] res = new int[k];
    int index = 0;
    for (int i = nums.length; i > 0; i--) {
        if (countArray[i] != null) {
            for (int num : countArray[i]) {
                res[index++] = num;
                if (index == k) return res;
            }
        }

    }
    return res;
}

 /**
 * Time Complexity: O(n)
 * Space Complexity: O(n)
 * use result int[] to save the answer of prefix and postfix for that position
 */
public int[] productExceptSelf(int[] nums) {
    if (nums == null) return new int[]{};
    int[] res = new int[nums.length];
    int prefix = 1, postfix = 1;

    for (int i = 0; i < nums.length; i++) {
        res[i] = prefix;
        prefix *= nums[i];
    }

    for (int i = nums.length - 1; i >= 0; i--) {
        res[i] *= postfix;
        postfix *= nums[i];
    }
    return res;
}
