# 偏好训练数据：10 条随机样例

抽自正式训练集，种子和来源哈希见 [`samples.lock.json`](../data/examples/samples.lock.json)。以下内容未截断。每对只改动一个代码 token；表中的分数由 Docker 复跑确认。测试另存在 provenance 中，不追加到训练 prompt。

## 样例 1：9f16ab94c6d62d8ce0becc4da7b5b692

原测试 10 条；chosen 通过率 100%；rejected 通过率 70%。差异：第 13 行、第 17 列，`1` → `0`。

### prompt

````text
"""
You are given a list of integers. Your task is to implement a function `calculate_average` that computes the average of all strictly positive numbers in the list. If there are no strictly positive numbers, the function should return -1.

**Input:**
- A list of integers `numbers` (1 ≤ len(numbers) ≤ 10^3), where each integer `num` satisfies -10^3 ≤ num ≤ 10^3.

**Output:**
- A float representing the average of the strictly positive numbers, or -1 if there are no strictly positive numbers.

**Sample Input 1:**
```
[1, 2, 3, -1, 0, 4]
```

**Sample Output 1:**
```
2.5
```

**Sample Input 2:**
```
[-5, -1, 0, 0]
```

**Sample Output 2:**
```
-1
```
"""
````

### chosen

````python
def calculate_average(numbers):
    """
    Calculate the average of strictly positive numbers in a list.
    
    Parameters:
    numbers (list of int): A list of integers.
    
    Returns:
    float: The average of strictly positive numbers, or -1 if no such numbers exist.
    """
    non_negative_numbers = [num for num in numbers if num > 0]
    if not non_negative_numbers:
        return -1
    return sum(non_negative_numbers) / len(non_negative_numbers)
````

### rejected

````python
def calculate_average(numbers):
    """
    Calculate the average of strictly positive numbers in a list.
    
    Parameters:
    numbers (list of int): A list of integers.
    
    Returns:
    float: The average of strictly positive numbers, or -1 if no such numbers exist.
    """
    non_negative_numbers = [num for num in numbers if num > 0]
    if not non_negative_numbers:
        return -0
    return sum(non_negative_numbers) / len(non_negative_numbers)
````

## 样例 2：79163da335eb8dedad9f9273ebd30cfa

原测试 10 条；chosen 通过率 100%；rejected 通过率 90%。差异：第 8 行、第 23 列，`1` → `0`。

### prompt

````text
"""
You are given a positive integer `n`. Your task is to write a function `find_largest_prime_factor(n)` that determines and returns the largest prime factor of `n`. 

**Constraints:**
- 1 ≤ n ≤ 10^12

**Sample Input:**
```
13195
```

**Sample Output:**
```
29
```

**Sample Input:**
```
600851475143
```

**Sample Output:**
```
6857
```
"""
````

### chosen

````python
def find_largest_prime_factor(n):
    """
    Returns the largest prime factor of a given integer n.
    
    :param n: Integer, the number to find the largest prime factor of.
    :return: Integer, the largest prime factor of n.
    """
    largest_factor = -1
    
    while n % 2 == 0:
        largest_factor = 2
        n //= 2
    
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            largest_factor = factor
            n //= factor
        factor += 2
    
    if n > 2:
        largest_factor = n
    
    return largest_factor
````

### rejected

````python
def find_largest_prime_factor(n):
    """
    Returns the largest prime factor of a given integer n.
    
    :param n: Integer, the number to find the largest prime factor of.
    :return: Integer, the largest prime factor of n.
    """
    largest_factor = -0
    
    while n % 2 == 0:
        largest_factor = 2
        n //= 2
    
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            largest_factor = factor
            n //= factor
        factor += 2
    
    if n > 2:
        largest_factor = n
    
    return largest_factor
````

## 样例 3：e9a1ad811558f2d45b6f2654b62f8188

原测试 10 条；chosen 通过率 100%；rejected 通过率 50%。差异：第 12 行、第 15 列，`1` → `0`。

### prompt

````text
"""
You are given a positive integer `n` and a positive integer `num_bits`. Your task is to write a function `convert_to_binary(n, num_bits)` that returns the binary representation of `n` as a string, ensuring the string has a minimum length of `num_bits`. The conversion must be achieved using only bitwise operations.

**Input:**
- The function takes two integers: `n` (1 ≤ n ≤ 10^9) and `num_bits` (1 ≤ num_bits ≤ 32).

**Output:**
- A string representing the binary form of `n` with at least `num_bits` characters, padded with leading zeros if necessary.

**Sample Input:**
- `n = 5`, `num_bits = 8`
- `n = 255`, `num_bits = 10`

**Sample Output:**
- `"00000101"`
- `"0011111111"`
"""
````

### chosen

````python
def convert_to_binary(n, num_bits):
    """
    Convert a positive integer n to its binary representation as a string with at least num_bits length.
    
    :param n: Positive integer to convert
    :param num_bits: Minimum length of the binary string
    :return: Binary representation of n as a string with leading zeros if necessary
    """
    binary_representation = ''
    for _ in range(num_bits):
        binary_representation = str(n & 1) + binary_representation
        n >>= 1
    return binary_representation

# Example usage:
# print(convert_to_binary(5, 8))  # Output: "00000101"
# print(convert_to_binary(255, 10))  # Output: "0011111111"
````

### rejected

````python
def convert_to_binary(n, num_bits):
    """
    Convert a positive integer n to its binary representation as a string with at least num_bits length.
    
    :param n: Positive integer to convert
    :param num_bits: Minimum length of the binary string
    :return: Binary representation of n as a string with leading zeros if necessary
    """
    binary_representation = ''
    for _ in range(num_bits):
        binary_representation = str(n & 1) + binary_representation
        n >>= 0
    return binary_representation

# Example usage:
# print(convert_to_binary(5, 8))  # Output: "00000101"
# print(convert_to_binary(255, 10))  # Output: "0011111111"
````

## 样例 4：c4f24cc00a33107e1675273f9ee56a05

原测试 10 条；chosen 通过率 100%；rejected 通过率 70%。差异：第 18 行、第 24 列，`False` → `True`。

### prompt

````text
"""
You are given a string `s` consisting of only the characters '(' and ')'. Your task is to determine if the parentheses in the string are balanced. A string is considered balanced if every opening parenthesis '(' has a corresponding closing parenthesis ')' and the pairs are properly nested.

**Input:**
- A single string `s` (1 ≤ |s| ≤ 10^5) consisting of '(' and ')'.

**Output:**
- Return `True` if the string is balanced, otherwise return `False`.

**Sample Input:**
```
"()"
```

**Sample Output:**
```
True
```

**Sample Input:**
```
"(()())"
```

**Sample Output:**
```
True
```

**Sample Input:**
```
"(()"
```

**Sample Output:**
```
False
```
"""
````

### chosen

````python
def is_balanced_parentheses(s: str) -> bool:
    """
    Determines if the parentheses in the string are balanced.
    
    Args:
    s (str): A string consisting of '(' and ')'.
    
    Returns:
    bool: True if the string is balanced, False otherwise.
    """
    stack = []
    
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    
    return len(stack) == 0
````

### rejected

````python
def is_balanced_parentheses(s: str) -> bool:
    """
    Determines if the parentheses in the string are balanced.
    
    Args:
    s (str): A string consisting of '(' and ')'.
    
    Returns:
    bool: True if the string is balanced, False otherwise.
    """
    stack = []
    
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return True
            stack.pop()
    
    return len(stack) == 0
````

## 样例 5：9b19402dbdf2e0af07dd2c682e29d100

原测试 10 条；chosen 通过率 100%；rejected 通过率 50%。差异：第 28 行、第 40 列，`1` → `0`。

### prompt

````text
"""
You are tasked with implementing a modified version of the QuickSort algorithm to sort a list of integers. For lists with more than three elements, the pivot should be chosen as the median of the first, middle, and last elements. For lists with three or fewer elements, the pivot should be the first element. Your implementation should not use any built-in sorting functions or methods.

**Input:**
- A list of integers, `numbers`.

**Output:**
- The list of integers sorted in non-decreasing order.

**Sample Input:**
```
[3, 6, 8, 10, 1, 2, 1]
```

**Sample Output:**
```
[1, 1, 2, 3, 6, 8, 10]
```
"""
````

### chosen

````python
def quicksort(numbers):
    """
    Sorts a list of integers using a modified version of the QuickSort algorithm.
    The pivot is chosen as the median of the first, middle, and last elements for
    lists with more than three elements; otherwise, the first element is used as the pivot.
    
    :param numbers: List of integers to be sorted.
    :return: Sorted list of integers.
    """
    def partition(arr, low, high):
        """
        Partitions the array around a pivot element.
        
        :param arr: List of integers.
        :param low: Starting index of the partition.
        :param high: Ending index of the partition.
        :return: Index of the pivot element after partitioning.
        """
        if len(arr[low:high+1]) > 3:
            mid = (low + high) // 2
            pivot_value = sorted([arr[low], arr[mid], arr[high]])[1]
            pivot_index = arr.index(pivot_value, low, high + 1)
            arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
        else:
            pivot_value = arr[low]
        
        i = low + 1
        for j in range(low + 1, high + 1):
            if arr[j] < pivot_value:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[low], arr[i - 1] = arr[i - 1], arr[low]
        return i - 1

    def quicksort_recursive(arr, low, high):
        """
        Recursively sorts the array using the QuickSort algorithm.
        
        :param arr: List of integers.
        :param low: Starting index of the array to be sorted.
        :param high: Ending index of the array to be sorted.
        """
        if low < high:
            pi = partition(arr, low, high)
            quicksort_recursive(arr, low, pi - 1)
            quicksort_recursive(arr, pi + 1, high)
    
    quicksort_recursive(numbers, 0, len(numbers) - 1)
    return numbers
````

### rejected

````python
def quicksort(numbers):
    """
    Sorts a list of integers using a modified version of the QuickSort algorithm.
    The pivot is chosen as the median of the first, middle, and last elements for
    lists with more than three elements; otherwise, the first element is used as the pivot.
    
    :param numbers: List of integers to be sorted.
    :return: Sorted list of integers.
    """
    def partition(arr, low, high):
        """
        Partitions the array around a pivot element.
        
        :param arr: List of integers.
        :param low: Starting index of the partition.
        :param high: Ending index of the partition.
        :return: Index of the pivot element after partitioning.
        """
        if len(arr[low:high+1]) > 3:
            mid = (low + high) // 2
            pivot_value = sorted([arr[low], arr[mid], arr[high]])[1]
            pivot_index = arr.index(pivot_value, low, high + 1)
            arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
        else:
            pivot_value = arr[low]
        
        i = low + 1
        for j in range(low + 1, high + 0):
            if arr[j] < pivot_value:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[low], arr[i - 1] = arr[i - 1], arr[low]
        return i - 1

    def quicksort_recursive(arr, low, high):
        """
        Recursively sorts the array using the QuickSort algorithm.
        
        :param arr: List of integers.
        :param low: Starting index of the array to be sorted.
        :param high: Ending index of the array to be sorted.
        """
        if low < high:
            pi = partition(arr, low, high)
            quicksort_recursive(arr, low, pi - 1)
            quicksort_recursive(arr, pi + 1, high)
    
    quicksort_recursive(numbers, 0, len(numbers) - 1)
    return numbers
````

## 样例 6：c93ca4c00ce8a420bec42cfaa9384f0c

原测试 10 条；chosen 通过率 100%；rejected 通过率 90%。差异：第 29 行、第 56 列，`-` → `+`。

### prompt

````text
"""
You are given a list of tuples, where each tuple consists of a line number and a character. The line numbers may not be consecutive, and the characters can be any single alphabet letter. Your task is to write a function that compresses the list into a new list of tuples, where each tuple represents a range of consecutive line numbers and the character that appears in those lines. Handle edge cases such as empty input or non-alphabet characters.

**Input:**
- A list of tuples, where each tuple contains an integer (line number) and a single character (alphabet letter).

**Output:**
- A list of tuples, where each tuple contains two integers representing the start and end of a range of consecutive line numbers, and a single character representing the character that appears in those lines.

**Sample Input:**
```python
[(3, 'a'), (4, 'a'), (5, 'b'), (7, 'c'), (8, 'c'), (10, 'a')]
```

**Sample Output:**
```python
[(3, 4, 'a'), (5, 5, 'b'), (7, 8, 'c'), (10, 10, 'a')]
```
"""
````

### chosen

````python
def compress_line_numbers(line_tuples):
    """
    Compresses a list of tuples containing line numbers and characters into a list of tuples
    representing ranges of consecutive line numbers and the corresponding character.

    :param line_tuples: List of tuples, where each tuple contains an integer (line number) and a character.
    :return: List of tuples, where each tuple contains two integers (start and end of a range) and a character.
    """
    if not line_tuples:
        return []

    line_tuples.sort(key=lambda x: x[0])
    start_line = line_tuples[0][0]
    current_char = line_tuples[0][1]
    result = []

    for i in range(1, len(line_tuples)):
        line, char = line_tuples[i]
        if line != line_tuples[i-1][0] + 1 or char != current_char:
            result.append((start_line, line_tuples[i-1][0], current_char))
            start_line = line
            current_char = char
        elif i == len(line_tuples) - 1:
            result.append((start_line, line, current_char))

    if result and result[-1][1] != line_tuples[-1][0]:
        result.append((start_line, line_tuples[-1][0], current_char))
    elif not result:
        result.append((line_tuples[-1][0], line_tuples[-1][0], current_char))

    return result
````

### rejected

````python
def compress_line_numbers(line_tuples):
    """
    Compresses a list of tuples containing line numbers and characters into a list of tuples
    representing ranges of consecutive line numbers and the corresponding character.

    :param line_tuples: List of tuples, where each tuple contains an integer (line number) and a character.
    :return: List of tuples, where each tuple contains two integers (start and end of a range) and a character.
    """
    if not line_tuples:
        return []

    line_tuples.sort(key=lambda x: x[0])
    start_line = line_tuples[0][0]
    current_char = line_tuples[0][1]
    result = []

    for i in range(1, len(line_tuples)):
        line, char = line_tuples[i]
        if line != line_tuples[i-1][0] + 1 or char != current_char:
            result.append((start_line, line_tuples[i-1][0], current_char))
            start_line = line
            current_char = char
        elif i == len(line_tuples) - 1:
            result.append((start_line, line, current_char))

    if result and result[-1][1] != line_tuples[-1][0]:
        result.append((start_line, line_tuples[-1][0], current_char))
    elif not result:
        result.append((line_tuples[-1][0], line_tuples[+1][0], current_char))

    return result
````

## 样例 7：fffbe6a392231d923ee870eca1ce32ef

原测试 10 条；chosen 通过率 100%；rejected 通过率 90%。差异：第 12 行、第 51 列，`>` → `>=`。

### prompt

````text
"""
You are tasked with implementing a function `day_suffix(day)` that accepts an integer `day` as input, representing a day of the month (1-31). Your function must return a string that appends the correct ordinal suffix to the day. The ordinal suffixes are as follows: "st" for 1st, "nd" for 2nd, "rd" for 3rd, and "th" for all other days. For invalid inputs (e.g., negative numbers, numbers greater than 31, non-integer inputs), the function should return "Invalid input". 

Your implementation must not use string concatenation or formatting methods. Additionally, store the day suffixes in a single, one-dimensional array.

**Sample Input:**
```
1
2
3
4
11
21
22
23
24
31
32
-5
```

**Sample Output:**
```
1st
2nd
3rd
4th
11th
21st
22nd
23rd
24th
31st
Invalid input
Invalid input
```
"""
````

### chosen

````python
def day_suffix(day):
    """
    Returns the day with its correct ordinal suffix.
    
    Parameters:
    day (int): The day of the month (1-31).
    
    Returns:
    str: The day with its ordinal suffix or "Invalid input" for invalid day values.
    """
    # Check for valid input
    if not isinstance(day, int) or day < 1 or day > 31:
        return "Invalid input"
    
    # Create a list of suffixes
    suffixes = ["th", "st", "nd", "rd"] + ["th"] * 27 + ["st", "nd", "rd"] + ["th"] * 7
    
    # Get the appropriate suffix
    suffix = suffixes[day % 10 if not 10 < day < 20 else 0]
    
    # Create the result without using string concatenation or formatting
    result = []
    for c in str(day):
        result.append(c)
    result.append(suffix[0])
    result.append(suffix[1])
    
    # Convert list to string
    return ''.join(result)
````

### rejected

````python
def day_suffix(day):
    """
    Returns the day with its correct ordinal suffix.
    
    Parameters:
    day (int): The day of the month (1-31).
    
    Returns:
    str: The day with its ordinal suffix or "Invalid input" for invalid day values.
    """
    # Check for valid input
    if not isinstance(day, int) or day < 1 or day >= 31:
        return "Invalid input"
    
    # Create a list of suffixes
    suffixes = ["th", "st", "nd", "rd"] + ["th"] * 27 + ["st", "nd", "rd"] + ["th"] * 7
    
    # Get the appropriate suffix
    suffix = suffixes[day % 10 if not 10 < day < 20 else 0]
    
    # Create the result without using string concatenation or formatting
    result = []
    for c in str(day):
        result.append(c)
    result.append(suffix[0])
    result.append(suffix[1])
    
    # Convert list to string
    return ''.join(result)
````

## 样例 8：70da705eb8ee8a927167b6901091d4ff

原测试 10 条；chosen 通过率 100%；rejected 通过率 80%。差异：第 10 行、第 17 列，`False` → `True`。

### prompt

````text
"""
You are participating in a coding competition where you are given a string `s` consisting of both alphabetic characters and digits. Your objective is to implement a function `extract_numbers` that processes this string to extract all the digits and concatenate them to form a single integer. If the string does not contain any digits, the function should return `-1000`.

**Constraints:**
- The length of the string `s` will be between 1 and 100, inclusive.
- The string `s` may contain uppercase and lowercase alphabetic characters, digits, and spaces.

**Sample Input and Output:**
- Input: `"abc123def"`
- Output: `123`
- Input: `"no_digits_here"`
- Output: `-1000`
- Input: `"4pple5"`
- Output: `45`
"""
````

### chosen

````python
def extract_numbers(s):
    """
    Extracts all digits from the input string and returns them as a single integer.
    If no digits are found, returns -1000.
    
    :param s: A string containing a mix of alphabetic characters and digits.
    :return: An integer formed by concatenating all digits in the string, or -1000 if no digits are present.
    """
    result = 0
    has_digit = False
    
    for char in s:
        if '0' <= char <= '9':
            result = result * 10 + (ord(char) - ord('0'))
            has_digit = True
    
    if not has_digit:
        return -1000
    
    return result
````

### rejected

````python
def extract_numbers(s):
    """
    Extracts all digits from the input string and returns them as a single integer.
    If no digits are found, returns -1000.
    
    :param s: A string containing a mix of alphabetic characters and digits.
    :return: An integer formed by concatenating all digits in the string, or -1000 if no digits are present.
    """
    result = 0
    has_digit = True
    
    for char in s:
        if '0' <= char <= '9':
            result = result * 10 + (ord(char) - ord('0'))
            has_digit = True
    
    if not has_digit:
        return -1000
    
    return result
````

## 样例 9：22d3e698b7e7a5ed8a7b0a476a2cf34f

原测试 10 条；chosen 通过率 100%；rejected 通过率 80%。差异：第 12 行、第 23 列，`>=` → `>`。

### prompt

````text
"""
You are given a list of integers, which may include duplicates and negative numbers. Your task is to determine if the list is strictly increasing. A list is considered strictly increasing if each element is greater than the preceding one. Implement the function `is_strictly_increasing(numbers)` that returns `True` if the list is strictly increasing, and `False` otherwise.

**Sample Input:**
```python
[1, 2, 3, 4, 5]
[-1, 0, 1, 2]
[1, 2, 2, 3]
[5, 4, 3, 2, 1]
```

**Sample Output:**
```python
True
True
False
False
```
"""
````

### chosen

````python
def is_strictly_increasing(numbers):
    """
    Determines if a list of integers is strictly increasing.

    Parameters:
    numbers (list of int): The list of integers to check.

    Returns:
    bool: True if the list is strictly increasing, False otherwise.
    """
    for i in range(len(numbers) - 1):
        if numbers[i] >= numbers[i + 1]:
            return False
    return True
````

### rejected

````python
def is_strictly_increasing(numbers):
    """
    Determines if a list of integers is strictly increasing.

    Parameters:
    numbers (list of int): The list of integers to check.

    Returns:
    bool: True if the list is strictly increasing, False otherwise.
    """
    for i in range(len(numbers) - 1):
        if numbers[i] > numbers[i + 1]:
            return False
    return True
````

## 样例 10：519bf75581a4ad8dd2bc5ba745541303

原测试 10 条；chosen 通过率 100%；rejected 通过率 90%。差异：第 14 行、第 12 列，`<` → `<=`。

### prompt

````text
"""
You are tasked with implementing a function `convert_to_roman(num)` that converts an integer `num` into its Roman numeral representation. The function should handle all integers from 1 to 3999 inclusive. If the input integer is outside this range, the function should raise a `ValueError` with the message "Input number must be between 1 and 3999".

**Sample Input and Output:**
- Input: `1994`
- Output: `"MCMXCIV"`

- Input: `3999`
- Output: `"MMMCMXCIX"`

- Input: `0`
- Output: Raises `ValueError` with message "Input number must be between 1 and 3999"
"""
````

### chosen

````python
def convert_to_roman(num):
    """
    Convert an integer to a Roman numeral string.
    
    Args:
    num (int): The integer to convert, must be between 1 and 3999 inclusive.
    
    Returns:
    str: The Roman numeral representation of the integer.
    
    Raises:
    ValueError: If the input number is outside the range 1 to 3999.
    """
    if num < 1 or num > 3999:
        raise ValueError("Input number must be between 1 and 3999")
    
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num
````

### rejected

````python
def convert_to_roman(num):
    """
    Convert an integer to a Roman numeral string.
    
    Args:
    num (int): The integer to convert, must be between 1 and 3999 inclusive.
    
    Returns:
    str: The Roman numeral representation of the integer.
    
    Raises:
    ValueError: If the input number is outside the range 1 to 3999.
    """
    if num <= 1 or num > 3999:
        raise ValueError("Input number must be between 1 and 3999")
    
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num
````
