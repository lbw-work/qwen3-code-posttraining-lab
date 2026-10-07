# GRPO：10 条完整随机样例

来自最终 2,005 道冻结训练题；来源哈希见 `data/examples/samples.lock.json`。

## 1. `levenshtein_distance`

来源 ID：`2c5cd3a79b0dcfa8a6261c9c4cfe1ad0`

模型输入：

```text
"""
You are given two strings, `s1` and `s2`. Your task is to implement a function that calculates the Levenshtein distance between these two strings. The Levenshtein distance is defined as the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into the other.

**Input:**
- Two strings `s1` and `s2` (1 ≤ |s1|, |s2| ≤ 100).

**Output:**
- An integer representing the Levenshtein distance between `s1` and `s2`.

**Sample Input:**
```
kitten
sitting
```

**Sample Output:**
```
3
```
"""

```

奖励测试：

```python
assert levenshtein_distance("", "") == 0
assert levenshtein_distance("a", "") == 1
assert levenshtein_distance("", "a") == 1
assert levenshtein_distance("abc", "abc") == 0
assert levenshtein_distance("abc", "abcd") == 1
assert levenshtein_distance("kitten", "sitting") == 3
assert levenshtein_distance("flaw", "lawn") == 2
assert levenshtein_distance("intention", "execution") == 5
assert levenshtein_distance("algorithm", "altruistic") == 6
assert levenshtein_distance("karolin", "kathrin") == 3
```

## 2. `find_min_max`

来源 ID：`a523210d23d978e78a58aa2bab922dac`

模型输入：

```text
"""
You are tasked with implementing a function `find_min_max` that accepts a variable number of numeric arguments. Your function should determine and return a tuple containing the smallest and largest values among all the provided arguments. If no arguments are given, your function should raise a `ValueError` with the message "At least one argument is required".

**Sample Input:**
```python
find_min_max(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5)
```

**Sample Output:**
```python
(1, 9)
```

**Sample Input:**
```python
find_min_max(-7, -1, -5, -3)
```

**Sample Output:**
```python
(-7, -1)
```
"""

```

奖励测试：

```python
assert find_min_max(1, 2, 3, 4) == (1, 4)
assert find_min_max(-1, -2, -3, -4) == (-4, -1)
assert find_min_max(10, 20, 30, 40, 50) == (10, 50)
assert find_min_max(5) == (5, 5)
assert find_min_max(3, 3, 3, 3) == (3, 3)
assert find_min_max(100, 200, 50, 75) == (50, 200)
assert find_min_max(-10, 0, 10, 20) == (-10, 20)
assert find_min_max(1.5, 2.5, 3.5, 4.5) == (1.5, 4.5)
assert find_min_max(-5.5, -4.5, -3.5, -2.5) == (-5.5, -2.5)
assert find_min_max(0, 0, 0, 0, 0) == (0, 0)
```

## 3. `convert_list_to_dict`

来源 ID：`fb706c11d1aa479c69a77f09451399e1`

模型输入：

```text
"""
You are given a list of tuples, where each tuple consists of two elements: a key and a value. Your task is to implement a function `convert_list_to_dict` that converts this list of tuples into a dictionary. If there are duplicate keys in the list, the value corresponding to the last occurrence of the key should be retained in the resulting dictionary.

**Input:**
- A list of tuples, where each tuple contains two elements: a key and a value. The keys are unique strings, and the values are integers.

**Output:**
- A dictionary with keys and values as specified in the input list. If there are duplicate keys, the value of the last occurrence of the key should be used.

**Sample Input:**
```python
[('a', 1), ('b', 2), ('c', 3)]
```

**Sample Output:**
```python
{'a': 1, 'b': 2, 'c': 3}
```

**Sample Input:**
```python
[('a', 1), ('b', 2), ('a', 3)]
```

**Sample Output:**
```python
{'a': 3, 'b': 2}
```
"""

```

奖励测试：

```python
assert convert_list_to_dict([('a', 1), ('b', 2), ('c', 3)]) == {'a': 1, 'b': 2, 'c': 3}
assert convert_list_to_dict([('a', 1), ('b', 2), ('a', 3)]) == {'a': 3, 'b': 2}
assert convert_list_to_dict([('x', 10), ('y', 20), ('z', 30), ('x', 40)]) == {'x': 40, 'y': 20, 'z': 30}
assert convert_list_to_dict([]) == {}
assert convert_list_to_dict([('single', 'value')]) == {'single': 'value'}
assert convert_list_to_dict([('key1', 'value1'), ('key2', 'value2'), ('key3', 'value3'), ('key1', 'value4')]) == {'key1': 'value4', 'key2': 'value2', 'key3': 'value3'}
assert convert_list_to_dict([('duplicate', 1), ('duplicate', 2), ('duplicate', 3)]) == {'duplicate': 3}
assert convert_list_to_dict([('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]) == {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
assert convert_list_to_dict([('same', 'first'), ('same', 'second'), ('same', 'third')]) == {'same': 'third'}
assert convert_list_to_dict([('a', 1), ('b', 2), ('c', 3), ('a', 1), ('b', 2), ('c', 3)]) == {'a': 1, 'b': 2, 'c': 3}
```

## 4. `find_unique_elements`

来源 ID：`c702e99329792f6bb90a69dbe985cf66`

模型输入：

```text
"""
You are given two lists of unique integers, `list1` and `list2`. Your task is to implement a function `find_unique_elements` that returns a list of all unique elements from both input lists. The elements should be sorted in ascending order based on the index of their first appearance in `list1` and then `list2`. If an element appears in both lists, it should only appear once in the result.

**Input:**
- Two lists of unique integers, `list1` and `list2`.

**Output:**
- A list of unique integers sorted by their first appearance index in `list1` and then `list2`.

**Sample Input:**
- `list1 = [4, 5, 6]`
- `list2 = [6, 7, 8, 5]`

**Sample Output:**
- `[4, 5, 6, 7, 8]`
"""

```

奖励测试：

```python
assert find_unique_elements([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]
assert find_unique_elements([4, 5, 6], [6, 7, 8, 5]) == [4, 5, 6, 7, 8]
assert find_unique_elements([], [1, 2, 3]) == [1, 2, 3]
assert find_unique_elements([1, 2, 3], []) == [1, 2, 3]
assert find_unique_elements([], []) == []
assert find_unique_elements([10, 20, 30], [30, 20, 10]) == [10, 20, 30]
assert find_unique_elements([1, 3, 5], [2, 4, 6]) == [1, 3, 5, 2, 4, 6]
assert find_unique_elements([5, 10, 15], [10, 20, 25, 5]) == [5, 10, 15, 20, 25]
assert find_unique_elements([1, 2], [2, 1, 3, 4]) == [1, 2, 3, 4]
assert find_unique_elements([7, 8, 9], [9, 8, 7, 6, 5]) == [7, 8, 9, 6, 5]
```

## 5. `sum_values`

来源 ID：`f52e8ce3cb3b9aa5a6955b0e9d0b9959`

模型输入：

```text
"""
You are given a dictionary `input_dict` where each key is a string and each value is a list of integers. Your task is to implement a function `sum_values` that computes a new dictionary. This new dictionary should have the same keys as `input_dict`, but each key should map to the sum of the integers in the corresponding list from `input_dict`.

**Input:**
- A dictionary `input_dict` with string keys and list of integers as values.

**Output:**
- A dictionary with the same keys as `input_dict`, where each key maps to the sum of the integers in the corresponding list.

**Sample Input:**
```python
{"a": [1, 2, 3], "b": [4, 5], "c": [6]}
```

**Sample Output:**
```python
{'a': 6, 'b': 9, 'c': 6}
```
"""

```

奖励测试：

```python
assert sum_values({"a": [1, 2, 3], "b": [4, 5], "c": [6]}) == {"a": 6, "b": 9, "c": 6}
assert sum_values({"x": [10, 20, 30], "y": [5, 5, 5], "z": [1]}) == {"x": 60, "y": 15, "z": 1}
assert sum_values({"single": [42]}) == {"single": 42}
assert sum_values({"empty": []}) == {"empty": 0}
assert sum_values({}) == {}
assert sum_values({"negative": [-1, -2, -3], "positive": [1, 2, 3]}) == {"negative": -6, "positive": 6}
assert sum_values({"mixed": [-1, 2, -3, 4]}) == {"mixed": 2}
assert sum_values({"zero": [0, 0, 0], "nonzero": [1, 2, 3]}) == {"zero": 0, "nonzero": 6}
assert sum_values({"repeated": [5, 5, 5, 5], "unique": [1, 2, 3, 4]}) == {"repeated": 20, "unique": 10}
assert sum_values({"large": [1000000, 2000000, 3000000]}) == {"large": 6000000}
assert sum_values({'a': [1, 2, 3], 'b': [4, 5], 'c': [6]}) == {'a': 6, 'b': 9, 'c': 6}
```

## 6. `format_file_size`

来源 ID：`8cb537f35987739234445921868146f4`

模型输入：

```text
"""
You are tasked with implementing a function `format_file_size` that converts a given file size from bytes into a human-readable format. The function should return the file size as a string, rounded to two decimal places, using the largest appropriate unit (B, KB, MB, GB, TB, or PB). The input will be a non-negative integer representing the file size in bytes. Your function should handle sizes up to and including petabytes.

**Input:**
- An integer `size_in_bytes` (0 ≤ size_in_bytes ≤ 10^18).

**Output:**
- A string representing the file size in the most appropriate unit, rounded to two decimal places.

**Sample Inputs and Outputs:**
- Input: `1500000`
  - Output: `"1.43 MB"`
- Input: `1234567890123`
  - Output: `"1.12 TB"`
- Input: `1234567890123456`
  - Output: `"1.10 PB"`
"""

```

奖励测试：

```python
assert format_file_size(0) == "0.00 B"
assert format_file_size(1023) == "1023.00 B"
assert format_file_size(1024) == "1.00 KB"
assert format_file_size(1500000) == "1.43 MB"
assert format_file_size(1073741824) == "1.00 GB"
assert format_file_size(1099511627776) == "1.00 TB"
assert format_file_size(1125899906842624) == "1.00 PB"
assert format_file_size(1234567890123) == "1.12 TB"
assert format_file_size(1234567890123456) == "1.10 PB"
assert format_file_size(1099511627776000) == "1000.00 TB"
```

## 7. `validate_position`

来源 ID：`eed7a6d2545284c5216c5c757f2fd259`

模型输入：

```text
"""
You are given a 3D grid with dimensions (X=100, Y=100, Z=100). Implement the function `validate_position` that checks if a given position is valid within this grid. A position is valid if it consists of three integer coordinates (x, y, z) where 0 ≤ x < 100, 0 ≤ y < 100, and 0 ≤ z < 100.

**Input:**
- A single argument `position`, which is a tuple of three integers representing the coordinates (x, y, z).

**Output:**
- Return `True` if the position is valid, otherwise return `False`.

**Sample Input:**
- `validate_position((50, 50, 50))`
- `validate_position((100, 50, 50))`
- `validate_position((-1, 50, 50))`
- `validate_position((50, 50, 100))`
- `validate_position((50, 50, 50, 50))`

**Sample Output:**
- `True`
- `False`
- `False`
- `False`
- `False`
"""

```

奖励测试：

```python
assert validate_position((0, 0, 0)) == True
assert validate_position((99, 99, 99)) == True
assert validate_position((100, 100, 100)) == False
assert validate_position((-1, 0, 0)) == False
assert validate_position((0, -1, 0)) == False
assert validate_position((0, 0, -1)) == False
assert validate_position((50, 50, 50)) == True
assert validate_position((100, 0, 0)) == False
assert validate_position((0, 100, 0)) == False
assert validate_position((0, 0, 100)) == False
```

## 8. `validate_user_profile`

来源 ID：`1dc7ced23224a8751d73b591cdc21949`

模型输入：

```text
"""
In a competitive programming challenge, you are required to implement a function `validate_user_profile` that verifies the integrity of a user profile represented as a dictionary. The user profile must satisfy the following criteria:

- It must contain exactly the keys `age`, `city`, and `country`.
- The value for the key `age` must be an integer.
- The values for the keys `city` and `country` must be strings.

Additionally, the function should accept an optional boolean parameter `check_for_name_too`. If `check_for_name_too` is `True`, the function must also ensure that the dictionary contains a key `name` with a string value. If the `name` key is absent or its value is not a string, the function should return `False`.

Your task is to implement the `validate_user_profile` function without using any conditional statements (such as `if` or `try-except`).

**Sample Input:**
```python
user_profile1 = {'age': 25, 'city': 'New York', 'country': 'USA'}
user_profile2 = {'age': 30, 'city': 'London', 'country': 'UK', 'name': 'Alice'}
user_profile3 = {'age': 'thirty', 'city': 'Paris', 'country': 'France'}
user_profile4 = {'age': 28, 'city': 'Berlin', 'country': 'Germany', 'name': 123}
```

**Sample Output:**
```python
validate_user_profile(user_profile1)  # True
validate_user_profile(user_profile2, check_for_name_too=True)  # True
validate_user_profile(user_profile3)  # False
validate_user_profile(user_profile4, check_for_name_too=True)  # False
```
"""

```

奖励测试：

```python
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 'USA'}) == True
assert validate_user_profile({'age': 25, 'city': 'New York'}) == False
assert validate_user_profile({'age': '25', 'city': 'New York', 'country': 'USA'}) == False
assert validate_user_profile({'age': 25, 'city': 123, 'country': 'USA'}) == False
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 123}) == False
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 'USA', 'name': 'John'}) == True
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 'USA', 'name': 123}, check_for_name_too=True) == False
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 'USA', 'name': 'John'}, check_for_name_too=False) == True
assert validate_user_profile({'age': 25, 'city': 'New York', 'country': 'USA', 'name': ''}, check_for_name_too=True) == True
assert validate_user_profile({}, check_for_name_too=True) == False
```

## 9. `calculate_2d_distance`

来源 ID：`ac5e6b2f8e1aeda372fce88425bb828b`

模型输入：

```text
"""
You are given two points in a 2D plane represented as tuples of two floating-point numbers. Your task is to write a function `calculate_2d_distance` that computes the Euclidean distance between these two points. The function should handle cases where the points are identical, resulting in a distance of zero.

**Input:**
- Two tuples, each containing two floating-point numbers representing the coordinates of the points.

**Output:**
- A floating-point number representing the Euclidean distance between the two points.

**Sample Input:**
- `((1.0, 2.0), (4.0, 6.0))`
- `((0.0, 0.0), (0.0, 0.0))`

**Sample Output:**
- `5.0`
- `0.0`
"""

```

奖励测试：

```python
assert calculate_2d_distance((1.0, 2.0), (4.0, 6.0)) == 5.0
assert calculate_2d_distance((0.0, 0.0), (0.0, 0.0)) == 0.0
assert calculate_2d_distance((3.0, 4.0), (0.0, 0.0)) == 5.0
assert calculate_2d_distance((-1.0, -1.0), (1.0, 1.0)) == 2.8284271247461903
assert calculate_2d_distance((5.0, 5.0), (5.0, 5.0)) == 0.0
assert calculate_2d_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
assert calculate_2d_distance((1.5, 2.5), (4.5, 6.5)) == 5.0
assert calculate_2d_distance((-3.0, -4.0), (3.0, 4.0)) == 10.0
assert calculate_2d_distance((10.0, 10.0), (10.0, 20.0)) == 10.0
assert calculate_2d_distance((0.0, 0.0), (-3.0, -4.0)) == 5.0
```

## 10. `event_frequency`

来源 ID：`dfdfcc101ccc2a69b40bdccc792fcf79`

模型输入：

```text
"""
You are given a string `event_string` representing a sequence of events separated by commas. Each event is a non-empty string of alphanumeric characters. Your task is to implement a function `event_frequency` that returns a dictionary where the keys are the unique events and the values are their respective frequencies in the sequence. If the input string is empty, the function should return an empty dictionary.

**Sample Input:**
```
"jump,run,jump"
```

**Sample Output:**
```
{'jump': 2, 'run': 1}
```

**Constraints:**
- The input string may be empty.
- Each event consists of alphanumeric characters only.
- The number of events in the input string does not exceed 1000.
"""

```

奖励测试：

```python
assert event_frequency("jump,run,jump") == {'jump': 2, 'run': 1}
assert event_frequency("walk,run,walk,run") == {'walk': 2, 'run': 2}
assert event_frequency("swim,bike,run,swim,bike,run") == {'swim': 2, 'bike': 2, 'run': 2}
assert event_frequency("jump") == {'jump': 1}
assert event_frequency("") == {}
assert event_frequency("a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z") == {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1, 'k': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 1, 'q': 1, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 1, 'w': 1, 'x': 1, 'y': 1, 'z': 1}
assert event_frequency("1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0") == {'1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2, '8': 2, '9': 2, '0': 2}
assert event_frequency("event1,event2,event3,event1,event2,event3,event1") == {'event1': 3, 'event2': 2, 'event3': 2}
assert event_frequency("single_event") == {'single_event': 1}
assert event_frequency("event_with_underscores,another_event") == {'event_with_underscores': 1, 'another_event': 1}
assert event_frequency('jump,run,jump') == {'jump': 2, 'run': 1}
```
