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


/**
* Reverse a linked list
*/
public ListNode reverseList(ListNode head) {
    ListNode curr = head;
    ListNode pre = null;
    ListNode next = null;

    while (curr != null) {
        next = curr.next;
        curr.next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}

public ListNode reverseList(ListNode head) {
    return rev(head, null);
}

public ListNode rev(ListNode node, ListNode pre) {
    if (node == null) return pre;
    ListNode temp = node.next;
    node.next = pre;
    return rev(temp, node);
}

/**
* Remove Nth node from end
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

/**
* find LCA in BST
* both greater/less than root, find on right/left branch, otherwise return root
*/
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
    if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
    return root;
}

/**
* rightSideView of tree
* 压入queue时先压入右node 这样可以把这个level的第一个node加入res
*/
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Queue<TreeNode> q = new LinkedList<>();
    if (root == null) return res;

    q.offer(root);

    while(!q.isEmpty()) {
        int size = q.size();
        for(int i = size; i > 0; i--) {
            TreeNode node = q.poll();
            if (i == size) res.add(node.val);
            if (node.right != null) q.offer(node.right);
            if (node.left != null) q.offer(node.left);
        }
    }
    return res;
}

/**
* 1448: 
* Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.
* Return the number of good nodes in the binary tree.
* pass down the max value
*/
public int goodNodes(TreeNode root) {
    return dfs(root, root.val);
}

private int dfs(TreeNode root, int maxVal) {
    if(root == null) return 0;
    int curr = root.val >= maxVal? 1 : 0;

    maxVal = Math.max(maxVal, root.val);
    int left = dfs(root.left, maxVal);
    int right = dfs(root.right, maxVal);
    return left + right + curr;
}

/**
* Use null as the initial max and min values as Integer.MAX_VALUE/Integer.MIN_VALUE have limits
*/
public boolean isValidBST(TreeNode root) {
    return dfs(root, null, null);
}

private boolean dfs(TreeNode root, Integer minimum, Integer maximum) {
    if (root == null) return true;
    if (minimum != null && root.val <= minimum || maximum != null && root.val >= maximum) return false;

    return dfs(root.left, minimum, root.val) && dfs(root.right, root.val, maximum);
}

/**
* Get inOrder travse and check if the numbers are continuously increaseing
*/
public boolean isValidBST(TreeNode root) {
    if(root == null){
        return true;
    }
    
    List<Integer> inOrder = new ArrayList<>();
    inOrderTravse(root, inOrder);
    
    for(int i = 0; i < inOrder.size() - 1; i++) {
        if(inOrder.get(i) >= inOrder.get(i+1)){
            return false;
        }
    }
    
    return true;
    
}

public void inOrderTravse(TreeNode root, List res) {
    if (root == null) {
        return;
    }
    
    if (root != null) {
        inOrderTravse(root.left, res);
        res.add(root.val);
        inOrderTravse(root.right, res);
    }
}

/**
* Find Kth smallest node in BST
* recursive & iterative
*/
public int kthSmallest(TreeNode root, int k) {
    List<Integer> list = new ArrayList<>();
    inorder(root, list);
    return list.get(k - 1);
}

private void inorder(TreeNode root, List<Integer> list) {
    if (root == null) return;

    inorder(root.left, list);
    list.add(root.val);
    inorder(root.right, list);
}

public int kthSmallest(TreeNode root, int k) {
    Stack<TreeNode> st = new Stack<>();
    int n = 0;

    TreeNode curr = root;

    while (curr != null || !st.empty()) {
        while (curr != null) {
            st.push(curr);
            curr = curr.left;
        }
        curr = st.pop();
        n++;
        if (n == k) return curr.val;
        curr = curr.right;
    } 
    return curr.val;
}


/**
* Given two integer arrays preorder and inorder
* construct and return the binary tree.
*/

public TreeNode buildTree(int[] preorder, int[] inorder) {
    if (preorder.length == 0 || inorder.length == 0) return null;
    
    TreeNode root = new TreeNode(preorder[0]);
    int mid = 0;
    for (int i = 0; i < inorder.length; i++) {
        if (preorder[0] == inorder[i]) mid = i;
    }

    root.left = buildTree(Arrays.copyOfRange(preorder, 1, mid + 1), Arrays.copyOfRange(inorder, 0, mid));
    root.right = buildTree(Arrays.copyOfRange(preorder, mid + 1, preorder.length), Arrays.copyOfRange(inorder, mid + 1, inorder.length));
    return root;
}
/**
* Without copy a sub array
*/
HashMap<Integer, Integer> inorderMap = new HashMap<>();
public TreeNode buildTree(int[] preorder, int[] inorder) {
    for(int i = 0; i < inorder.length; i++) {
        inorderMap.put(inorder[i], i);
    }
    return build(preorder, 0, 0, inorder.length - 1);
}

private TreeNode build(int[] preorder, int preorderIndex, int inorderLow, int inorderHigh) {
    if (preorderIndex > preorder.length - 1 || inorderLow < 0 || inorderHigh > inorderMap.size() -1 || inorderLow > inorderHigh) return null;
    int val = preorder[preorderIndex];
    TreeNode root = new TreeNode(val);
    int mid = inorderMap.get(val);
    root.left = build(preorder, preorderIndex + 1, inorderLow, mid - 1);
    root.right = build(preorder, preorderIndex + mid - inorderLow + 1, mid + 1, inorderHigh);
    return root;
}


/**
* Subsets I
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

/**
* Subsets II with duplicates and non-ordered array
*/
public List<List<Integer>> subsetsWithDup(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    dfs(nums, 0, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int[] nums, int start, List<Integer> curr, List<List<Integer>> res) {
    res.add(new ArrayList<>(curr));

    for (int i = start; i < nums.length; i++) {
        if (i != start && nums[i] == nums[i - 1]) continue;
        curr.add(nums[i]);
        dfs(nums, i + 1, curr, res);
        curr.remove(curr.size() - 1);
    }
}

/**
* CombinationSum I
*/
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    dfs(candidates, 0, target, new ArrayList<Integer>(), res);
    return res;
}

private void dfs(int[] candidates, int start, int target, List<Integer> curr, List<List<Integer>> res) {
    if (target == 0) {
        res.add(new ArrayList<>(curr));
        return;
    } 
    if (target < 0 || start > candidates.length - 1) {
        return;
    } 
    for (int i = start; i < candidates.length; i++) {
        curr.add(candidates[i]);
        dfs(candidates, i, target - candidates[i], curr, res);
        curr.remove(curr.size() - 1);
    }
}

/**
* CombinationSum II
* remove duplicates
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



