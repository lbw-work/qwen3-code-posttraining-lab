# DPO 逐题审阅

打开 index.html 查看并排代码；cases.json 保留原始证据及来源哈希。

所有人工分类为静态代码审阅，没有重跑测试。未知 token 数，不断言达到 512 上限。

| 题库 | 退化 | 改善 | 均通过 | 均失败 |
| --- | ---: | ---: | ---: | ---: |
| humaneval | 10 | 24 | 72 | 58 |
| mbpp | 26 | 17 | 211 | 124 |

## 逐题阅读索引

| 题目 | 变化 | 人工分类 | 说明 |
| --- | --- | --- | --- |
| [HumanEval/11](index.html#HumanEval-11) | 退化 | 语法与无关续写 | 目标 string_xor 的实现与 SFT 一样，但 DPO 继续生成多个无关函数，最后的三引号字符串未闭合，整个模块无法解析。可确认的是语法错误；是否达到 token 上限尚未核实。 |
| [HumanEval/28](index.html#HumanEval-28) | 退化 | 语法与无关续写 | concatenate 的主体仍是正确的 join；后面追加了 reverse、count_* 等无关函数，最终留下未闭合 docstring，导致整段 solution 语法失败。 |
| [HumanEval/36](index.html#HumanEval-36) | 退化 | 算法逻辑 | 题目要求统计数字 7 出现的次数。DPO 用是否包含 7 代替 count('7')，77 中两个 7 只计一次；已有失败输入 78 会包含 77。 |
| [HumanEval/49](index.html#HumanEval-49) | 退化 | 算法逻辑 | SFT 用 pow(2,n,p)。DPO 奇数分支把乘 2 写成乘 modp(2,p)，相当于乘 4；已有失败输入 (3,5)，并且递归实现更复杂。 |
| [HumanEval/80](index.html#HumanEval-80) | 退化 | 算法逻辑 | 三字符互异被改成仅禁止三个字符全部相同。两个字符重复而第三个不同会被漏掉；已有失败输入 aabb。 |
| [HumanEval/83](index.html#HumanEval-83) | 退化 | 算法逻辑 | DPO 把已通过评测的 2*9*10**(n-2) 改为 2*10**(n-1)，例如 n=2 从 18 变为 20。已有原始测试失败；这里依据评测约定比较，不把 SFT 通过当作题意唯一无歧义的证明。 |
| [HumanEval/84](index.html#HumanEval-84) | 退化 | 题意与规则 | 题意是十进制各位数字求和后转二进制。DPO 改为统计 N 的二进制 1 个数后再转二进制；例如 1000 的两种统计不同。 |
| [HumanEval/87](index.html#HumanEval-87) | 退化 | 算法逻辑 | SFT 按行升序、列降序排序原坐标。DPO 只按行排序，再用结果序号覆盖原行号，并用长度为 2 的坐标元组反算列号，破坏了坐标及顺序。 |
| [HumanEval/142](index.html#HumanEval-142) | 退化 | 算法逻辑 | DPO 虽然给 lst[index] 写入平方或立方，最后累加的却是修改前的局部变量 value，所以返回原始元素和；同时新增输入列表修改副作用。 |
| [HumanEval/155](index.html#HumanEval-155) | 退化 | 边界与返回约定 | DPO 改用 while num>0 拆位，输入 0 时循环不执行，返回 (0,0)，漏掉数字 0 本身是一个偶数位；已有失败输入 0。 |
| [Mbpp/71](index.html#Mbpp-71) | 退化 | 算法逻辑 | DPO 在 gap 仍大于 1 时也把 sorted 设为 True，无交换即可提前退出。已有输入 [5,15,37,25,79] 在 gap=3 的一轮无交换后仍未排序完成。base=fail、plus=pass 仍按项目约定算失败。 |
| [Mbpp/77](index.html#Mbpp-77) | 退化 | 边界与返回约定 | 从 n%11==0 改为逐字符转 int，负数中的 '-' 会触发转换错误；已有失败输入 -1212。奇偶位差仅与 0 比较也不同于检查差是否为 11 的倍数。 |
| [Mbpp/84](index.html#Mbpp-84) | 退化 | 算法逻辑 | DPO 改动 Newman-Conway 的初值和递推式：n=2 返回 2，递推也不再是原 DP 的两项索引形式。已有 n=10 失败；可能还有递归性能问题，但未重跑确认。 |
| [Mbpp/86](index.html#Mbpp-86) | 退化 | 算法逻辑 | 中心六边形数的 3*n*n-3*n+1 被改成 n*(2*n-1)，公式对应的序列不同；已有 n=10 失败。 |
| [Mbpp/89](index.html#Mbpp-89) | 退化 | 边界与返回约定 | DPO 新增 n<=1 时返回 0，改变了原 n-1 的规则；失败输入 -5 应按本评测约定处理，而不是自作限制。 |
| [Mbpp/100](index.html#Mbpp-100) | 退化 | 算法逻辑 | SFT 逐个寻找下一个回文数。DPO 只改中间位，没有保证回文、进位或严格大于输入；已有 99 失败，偶数位输入还有分支缺失导致隐式 None。 |
| [Mbpp/105](index.html#Mbpp-105) | 退化 | 入口函数名 | 题目入口是 count，DPO 改成 count_true_booleans，完整模块中没有 count 定义。函数主体的意图相近，但测试无法按要求调用。 |
| [Mbpp/109](index.html#Mbpp-109) | 退化 | 题意与规则 | DPO 把 SFT 的左旋改为右旋；公开题干没有写清方向，但已有新增测试使用了有方向差异的输入。应记录为方向与评测约定不符，不把题干当作毫无歧义。 |
| [Mbpp/120](index.html#Mbpp-120) | 退化 | 顶层断言 | 目标函数主体与 SFT 数学上等价。DPO 额外生成顶层 assert，把 [(10,20),(15,5),(7,14)] 的最大乘积写成 210，实际为 200。导入时错误断言会阻止测试执行；这是静态算术判断，未重新执行该程序。 |
| [Mbpp/142](index.html#Mbpp-142) | 退化 | 边界与返回约定 | SFT 用 zip 遍历公共长度。DPO 按 list1 全长索引其他列表，且长度检查使用链式 !=，既可能误拒绝不同长度，也可能越界；已有长度为 6、5、8 的失败输入。 |
| [Mbpp/160](index.html#Mbpp-160) | 退化 | 算法逻辑 | DPO 改成不正确的欧几里得式求解：用 n 替代系数参与递推，不能保证 ax+by=n；已有 (2,3,7) 失败，后续 x<0 分支还可能用已变为 0 的 b 作除数。 |
| [Mbpp/165](index.html#Mbpp-165) | 退化 | 题意与规则 | SFT 比较字符与对应位置的字母。DPO 仅检查是否是英文字母，实质变成字母计数；已有 xbcefg 失败。 |
| [Mbpp/283](index.html#Mbpp-283) | 退化 | 题意与规则 | 题目比较每个数字的出现频次与数字本身。DPO 改成与 number%10 比较，不再计算频次；已有 1234 失败。 |
| [Mbpp/465](index.html#Mbpp-465) | 退化 | 边界与返回约定 | DPO 除 None 外也删除空字符串。题干的 empty 较含糊，但评测要求保留空字符串，已有含空字符串的新增测试失败；这是额外过滤与评测约定冲突。 |
| [Mbpp/587](index.html#Mbpp-587) | 退化 | 入口函数名 | 题目入口 list_tuple 被改成 list_to_tuple；tuple 转换主体相同，缺少指定入口导致失败。 |
| [Mbpp/592](index.html#Mbpp-592) | 退化 | 算法逻辑 | 相邻二项式系数的乘积 comb(n,i)*comb(n,i+1) 被改为 comb(n,i)**2，求的是不同的和；已有 n=3 失败。 |
| [Mbpp/602](index.html#Mbpp-602) | 退化 | 边界与返回约定 | 找到重复字符时逻辑等价，但无重复时把 None 改成空字符串。已有 abc 与空字符串输入失败，说明返回约定被改变。 |
| [Mbpp/608](index.html#Mbpp-608) | 退化 | 算法逻辑 | SFT 使用 Bell 三角形。DPO 用前面 bell 值的直接求和，缺少 Bell 数所需的组合系数或正确三角递推；已有 n=3 失败。 |
| [Mbpp/611](index.html#Mbpp-611) | 退化 | 边界与返回约定 | DPO 新增 n<0 时返回 None，禁止 Python 支持的负索引。已有 n=-1 的新增测试失败，目标函数应取最后一列而非拒绝输入。 |
| [Mbpp/632](index.html#Mbpp-632) | 退化 | 入口函数名 | 题目入口 move_zero 被改为 move_zeroes；此外算法改为原地修改列表。当前最直接可核实的失败线索是完整模块缺少指定入口，不能据此先断言原地修改是本次失败原因。 |
| [Mbpp/633](index.html#Mbpp-633) | 退化 | 算法逻辑 | 每对数的 XOR 结果应累加。DPO 把 += 改成 ^=，从求和变成累积异或；已有 [5,9,7,6],4 失败。 |
| [Mbpp/638](index.html#Mbpp-638) | 退化 | 边界与返回约定 | DPO 新增 velocity<=0 时返回 None。已有 (0,0) 新增测试失败，原公式在这个输入有定义；限制条件没有来自当前题目的依据。 |
| [Mbpp/726](index.html#Mbpp-726) | 退化 | 语法错误 | DPO 将题目中的数学下标直接写成 t_{i+1}，这不是合法 Python 表达式；ast.parse 可确认语法失败。 |
| [Mbpp/767](index.html#Mbpp-767) | 退化 | 算法逻辑 | DPO 使用 set 只记录是否出现，丢失重复元素次数；四个 1、目标和为 2 时应有 6 对，代码逻辑只会计 3 对。已有该输入失败。 |
| [Mbpp/791](index.html#Mbpp-791) | 退化 | 题意与规则 | 题目要求删除嵌套 tuple 元素。DPO 改成递归展开嵌套 tuple，把应删除的内容保留下来；已有原始测试失败。失败 JSON 的数组表示不能用于推断原参数一定是 list。 |
| [Mbpp/808](index.html#Mbpp-808) | 退化 | 题意与规则 | 题目要求 k 是否为输入 tuple 的元素。DPO 把它理解为 tuple 列表，逐个执行 k in t；原始数字 tuple 会触发不可迭代元素错误，另有嵌套结构的新增测试失败。 |
| [HumanEval/0](index.html#HumanEval-0) | 改善 | 语法与无关续写 | SFT 的完整 solution 无法语法解析；DPO 可解析，目标函数使用两两距离比较并通过现有测试。改善至少包含模块可解析性，不能直接归因于更强算法能力。 |
| [HumanEval/4](index.html#HumanEval-4) | 改善 | 语法与无关续写 | SFT 有语法失败，DPO 的平均绝对偏差实现可解析并通过测试；DPO 仍追加无关统计函数，末尾 z_score 留下单独 std 表达式，说明通过并不代表消除了无关续写。 |
| [HumanEval/5](index.html#HumanEval-5) | 改善 | 算法逻辑 | SFT 没有保留中间所有元素；DPO 逐个插入分隔符并保留原元素顺序。 |
| [HumanEval/8](index.html#HumanEval-8) | 改善 | 算法逻辑 | SFT 递归乘积误用了递归返回的求和项；DPO 分别累加与累乘。 |
| [HumanEval/31](index.html#HumanEval-31) | 改善 | 边界与返回约定 | DPO 将 n==1 改为 n<=1，覆盖零和负数等非素数边界，并改用整数式试除条件。 |
| [HumanEval/35](index.html#HumanEval-35) | 改善 | 语法与无关续写 | SFT 完整 solution 缺少必要缩进块；DPO 完整实现最大值遍历并通过现有测试。 |
| [HumanEval/37](index.html#HumanEval-37) | 改善 | 题意与规则 | SFT 破坏输出结构并处理错位置；DPO 只排序偶数索引元素，其他位置保留。 |
| [HumanEval/38](index.html#HumanEval-38) | 改善 | 算法逻辑 | SFT 的 decode 不执行逆旋转；DPO 将每组三字符向右旋转以撤销 encode 的左旋。 |
| [HumanEval/53](index.html#HumanEval-53) | 改善 | 语法与无关续写 | SFT 留下未闭合字符串；DPO 的 add 及完整模块可解析，因此通过测试，但仍生成了很多未要求的函数。 |
| [HumanEval/55](index.html#HumanEval-55) | 改善 | 算法逻辑 | DPO 增加 n=0 边界并采用迭代 Fibonacci；SFT 的无缓存递归还存在明显性能风险。具体失败机制未重跑区分。 |
| [HumanEval/64](index.html#HumanEval-64) | 改善 | 题意与规则 | DPO 补上大小写处理及结尾 y/Y 规则，SFT 只统计小写 aeiou。 |
| [HumanEval/68](index.html#HumanEval-68) | 改善 | 边界与返回约定 | 没有偶数时，DPO 返回空列表，而非 SFT 的 [inf,-1]。 |
| [HumanEval/69](index.html#HumanEval-69) | 改善 | 题意与规则 | DPO 从较大候选开始检查出现次数，SFT 从小到大返回第一个，未满足寻找最大符合值的要求。 |
| [HumanEval/90](index.html#HumanEval-90) | 改善 | 边界与返回约定 | 没有第二个不同最小值时，DPO 把 inf 转为 None。 |
| [HumanEval/104](index.html#HumanEval-104) | 改善 | 语法错误 | SFT 括号不匹配，DPO 的奇数数字筛选表达式合法并通过测试。 |
| [HumanEval/109](index.html#HumanEval-109) | 改善 | 题意与规则 | SFT 只检查原数组是否已排序；DPO 检查循环移位后能否得到排序数组。 |
| [HumanEval/110](index.html#HumanEval-110) | 改善 | 算法逻辑 | DPO 比较两个列表的偶数总量与 lst1 长度，SFT 仅检查是否有整条列表全为偶数。 |
| [HumanEval/111](index.html#HumanEval-111) | 改善 | 边界与返回约定 | DPO 处理空文本并给 max 设置默认值；SFT 对空计数直接求 max。 |
| [HumanEval/119](index.html#HumanEval-119) | 改善 | 题意与规则 | DPO 检查两个字符串的两种连接顺序；SFT 只分别检查每段自身是否平衡。 |
| [HumanEval/125](index.html#HumanEval-125) | 改善 | 题意与规则 | DPO 检查字母在字母表的位置奇偶，SFT 检查字符在字符串中的索引，混淆了两种位置。 |
| [HumanEval/131](index.html#HumanEval-131) | 改善 | 边界与返回约定 | DPO 增加 has_odd，没有奇数数字时按要求返回 0；SFT 保留乘积初值 1。 |
| [HumanEval/144](index.html#HumanEval-144) | 改善 | 数值精度 | DPO 用整数乘积取模判定分数乘积是否为整数，替代 SFT 浮点除法后的相等比较。 |
| [HumanEval/150](index.html#HumanEval-150) | 改善 | 算法逻辑 | SFT 硬编码小于 100 的素数；DPO 使用一般试除算法，不局限于该列表。 |
| [HumanEval/151](index.html#HumanEval-151) | 改善 | 类型约定 | DPO 增加整数类型筛选，SFT 把符合数值条件的浮点元素也纳入平方和。 |
| [Mbpp/9](index.html#Mbpp-9) | 改善 | 边界与返回约定 | 没有更短周期时，DPO 返回完整长度 n；SFT 返回 0。 |
| [Mbpp/12](index.html#Mbpp-12) | 改善 | 题意与规则 | DPO 按行和排序行，SFT 排序每行内部元素。 |
| [Mbpp/67](index.html#Mbpp-67) | 改善 | 算法逻辑 | DPO 的 Bell 数递推加入二项式系数；SFT 递推求的是不同的序列。与 Mbpp/608 的退化并存，不能断言 DPO 一概改善或损伤 Bell 数题。 |
| [Mbpp/69](index.html#Mbpp-69) | 改善 | 题意与规则 | DPO 判断连续子列表，SFT 只检查各元素是否出现在主列表。 |
| [Mbpp/259](index.html#Mbpp-259) | 改善 | 题意与规则 | DPO 对内部 tuple 的对应元素取最大值；SFT 对整个 tuple 做字典序 max。 |
| [Mbpp/264](index.html#Mbpp-264) | 改善 | 算法逻辑 | DPO 使用前两年及以后分段换算，SFT 一律乘 7。 |
| [Mbpp/274](index.html#Mbpp-274) | 改善 | 性能 | DPO 用 math.comb 替代无缓存的二项式递归，避免大量重复计算；具体测试失败是否由超时造成未重新核实。 |
| [Mbpp/294](index.html#Mbpp-294) | 改善 | 类型约定 | DPO 筛出整数再取 max，SFT 用 int 作为比较 key，仍可能返回字符串或接受不符合要求的元素。 |
| [Mbpp/409](index.html#Mbpp-409) | 改善 | 顶层断言 | 两个目标函数主体相同。SFT 顶层自写断言把 [(10,20),(15,2),(5,10)] 的最小乘积写成 10，实际为 30；DPO 没有追加这些断言。静态算术支持改善来自删除干扰，而非函数算法提升。 |
| [Mbpp/410](index.html#Mbpp-410) | 改善 | 类型约定 | DPO 只对整数求 min，SFT 直接对异质列表求 min。 |
| [Mbpp/424](index.html#Mbpp-424) | 改善 | 边界与返回约定 | DPO 将结果类型从 tuple 改为 list，符合现有判分的返回约定。 |
| [Mbpp/579](index.html#Mbpp-579) | 改善 | 题意与规则 | DPO 求两个集合的对称差；SFT 只求第一个 tuple 中不属于第二个的元素。 |
| [Mbpp/637](index.html#Mbpp-637) | 改善 | 题意与规则 | DPO 检查两个金额相等，SFT 检查第一个大于第二个，与无盈利无亏损含义不同。 |
| [Mbpp/644](index.html#Mbpp-644) | 改善 | 算法逻辑 | DPO 反转前 k 项并保留后缀，SFT 使用 arr[k:] 两次，没有正确处理前缀。 |
| [Mbpp/743](index.html#Mbpp-743) | 改善 | 边界与返回约定 | DPO 在空列表时提前返回，避免 SFT 对 len(lst)=0 取模。 |
| [Mbpp/784](index.html#Mbpp-784) | 改善 | 题意与规则 | DPO 取第一个偶数与第一个奇数的乘积，SFT 求全部元素的乘积。 |
| [Mbpp/788](index.html#Mbpp-788) | 改善 | 参数顺序 | DPO 的参数顺序与题干示例中的 list,string 一致；SFT 写成 string,list，拼接方式也不符。 |
