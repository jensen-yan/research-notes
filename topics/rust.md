https://rust-lang.org

https://rust-book.cs.brown.edu/experiment-intro.html

三大特点
1. 性能
	1. 运行速度和C++ 一样快
	2. 内存效率高
	3. 无运行时和垃圾回收期
	4. 性能关键服务，在嵌入式设备运行
2. 可靠性
	1. 类型系统和所有权模型，保障内存安全和线程安全
	2. 编译时候能识别错误
3. 生产力
	1. 文档，编译器好

使用场景
1. 命令行cli
2. webAssembly 这个web 应用
3. 网络
4. 嵌入式

概念
1. cargo: 内置的依赖管理器和构建工具
2. rustfmt: 格式化工具

对于我，更关注概念学习，不太需要关注语法细节，毕竟有AI 写代码
所有权系统 = c++ RALL 默认unique owenership
数 Rust 代码：

| Rust 概念        | C++ 类比               | 真正要理解的点                            |
| -------------- | -------------------- | ---------------------------------- |
| ownership      | `unique_ptr` / RAII  | 每个值有唯一 owner，owner 离开作用域自动释放       |
| move           | `std::move` 之后原对象不可用 | Rust 对非 `Copy` 类型默认 move，不允许继续用旧变量 |
| borrow         | 引用/指针                | 借用只是临时访问权，不拥有资源                    |
| mutable borrow | `T&` 非 const 引用      | 同一时间只能有一个可变借用                      |
| lifetime       | 引用有效性范围              | 不是“延长生命周期”，而是证明引用不会悬垂              |

重点是：**Rust 的所有权不是为了“难为程序员”，而是为了把资源生命周期和别名关系变成编译器可检查的东西。**

cargo 代码包 叫 crates

1. cargo new  : git init, add cargo.toml
2. cargo build: get target: 
3. cargo run:   ./target/debug/hello_cargo
4. cargo check: 检查能否编译，不编译
5. cargo build --release: 优化代码，运行更快
不依赖OS!

```rust
use std::io;  // include

fn main() {
    let apples = 5;     // 变量默认不可变
    let mut guess = String::new();  // mut 变量可变化

    io::stdin()
        .read_line(&mut guess)  // $ 引用，不用内存复制，内容放入guess
        .expect("Failed to read line");  // read_line 返回Result 枚举值，Ok Err
        // Result.expect = Output if ok else string

    println!("You guessed: {guess}");// 花括号输出
}
```

cargo.lock 文件，一旦第一次用的某个版本，后续不再下载新版本，保证可重复构建
cargo update 能坚持新版本并更新lock

### 变量
1. 默认变量let 不可变，除非显示添加mut 指示后续能修改
2. 常量const 永远不可变, 可以任何作用域声明，只能是常量表达式而非运行时生成（可以有限计算60 * 60 ），默认大小
3. 后定义变量可以遮蔽前者，
4. ```rust
fn main() {
    let x = 5;
	    let x = x + 1;  // 6
    {
        let x = x * 2;  // 12
        println!("The value of x in the inner scope is: {x}");
    }  // 结束后遮蔽也结束
    println!("The value of x is: {x}");  // 出去后变回6
}

    let spaces = "   ";
    let spaces = spaces.len();  // 遮蔽改类型String -> i32


    let mut spaces = "   ";
    spaces = spaces.len();   // 类型变化，失败！

   ```
5. 遮蔽是创建新变量，只是复用同一个名称。 遮蔽还能改类型，而mut 不能改类型
6. rust 是静态类型，必须编译时候知道所有类型。
7. 标量类型
	1. 整数： i32, u32; isize, usize
	2. 浮点: f32
	3. bool:  let f: bool = false;
	4. 字符: let z: char = 'z';
8. 复合类型
	1. 元祖： 不同类型元素变成一起，固定长度
		1. let tup: (i32, f64, u8) = (500, 6.4, 1);
		2. let (x, y, z) = tup;
		3. let f = tup.0;   取出变量0
		4. tup.0 += 5;    可修改元素
	2. 数组： 相同类型元素，固定长度
		1. let a = [1,2,3]
		2. let a = [2; 3]  = [2, 2, 2], 重复3次
		3. let first = a[0];
9. 函数：
	1. 参数必须包括类型声明
	2. fu add(x: i32, y: i32);   add(1,2)
10. statements: 语句，执行某些操作但不返回值； let a = 5;
	1. 和C, ruby 不同，let x = (let y = 5); 会失败
11. expressions: 表达式，会计算一个结果值;  
	1. 5 + 6 是表达式
	2. 函数调用
	3. 调用宏
	4. 花括号创建的新作用域块
	5. 表达式不包括结尾分号，如果加上分号会变成语句！
	6. ```rust
fn main() {
    let y = {
        let x = 3;
        x + 1    // 这里没有分号
    };   // 方括号是表达式，赋值给y

    println!("The value of y is: {y}");
}

fn five() -> i32 { 5 }   // 函数直接返回值
let x = five()
	   ```
12. if 表达式， if 条件 { } else { },  条件: bool
13. ```rust
    
let x;
if cond {
  x = 1;
} else {
  x = 2;
}
// 两者等价，注意let 不需要mut 声明，可以确定x 只会赋值一次，if 只会走一个分支
let x = if cond {1} else { 2 }; 
    ```
12. 循环
	1. loop： 不断执行直到明确停止 break; 
		1. continue(跳过循环中剩下代码进入下一次循环)
		2. usage: 重试一个你知道可能会失败操作， break a; 可以退出并返回需要的值
		3. break 退出当前循环，return 退出当前函数
		4. ```rust
	fn main() {
    let mut count = 0;
    'counting_up: loop {
        println!("count = {count}");
        let mut remaining = 10;

        loop {
            println!("remaining = {remaining}");
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;  // 嵌套循环，跳出最外层循环counting_up 了
            }
            remaining -= 1;
        }

        count += 1;
    }
    println!("End count = {count}");
}
		   ```
	2. while 条件 {}  
	3. for: 最常用最安全，  for a in (1..4).rev();   3,2,1


### 所有权
最独特，能不需要垃圾回收下保证内存安全，关注 借用，切片，内存布局
安全 = 不存在未定义行为
```rust
fn read(y: bool) {
    if y {
        println!("y is true!");
    }
}

fn main() {
    read(x); // oh no! x isn't defined! 主要此时x 类型还不知道
    let x = true;
}
```
1. python 解释器可以在运行时发现不对，在x 定义之前就读取会有NameError异常，但是每次读取都要检查，运行开销太大
2. rust 运行前编译检查，确保先定义再使用。
3. 否则，未定义行为可能暂时不影响，但是未来可能出问题（C++ 通过santisizer 缓解）

![[Pasted image 20260525145328.png|310]]
1. 变量存在于frames 帧 ， frame = 单个作用域（例如函数） 内变量到值的映射
2. frames 被组织成当前调用的函数的栈
	eg: L2 中main frame 在plus_one frame 上面， 函数返回后后者被释放
3. 表达式expression 读取变量，值会从stack 中复制过来，浪费存储
4. 在堆heap 上分配数据，然后通过指针指向对象，减少数据移动，可以长期存在。
	1. ```rust
	   let a = Box::new([0; 10000])
	   let b = a;   // 指向同一个堆数据了， 这里所有权从a 转交给了b !!!, b 不被用了，就释放对应的堆
	   ```
5. rust 不允许手动管理内存！
	1. 栈帧由rust 自动管理，调用函数结束自动释放。
	2. 堆也自动释放（no free!）：如果某个变量拥有了一个box, 那么当rust 释放该变量的帧时候，rust 也释放该box 的堆内存。
6. 使用Vec, String, HashMap 使用Box 来容纳可变数量元素
7. 例如这里，所有权从frist, 到name, 到full 逐渐转移
8. ![[Pasted image 20260525150716.png|567]]
9. 变量移动后不能被使用
	1. 这里如果println!("{full}, {first}"); 会失败，first 作为String 类型，给别人后就不能访问了
	2. 移动堆数据原则：如果变量x 把堆数据所有权移动给变量y, 那么x 在移动后不能再被使用
	3. 克隆 可以再被使用，相当于复制了一个堆内容

所有权总结（本质是指针管理的规范）
1. 所有堆heap 数据智能恰好有一个变量拥有
2. 一旦heap 数据所有者超出作用域，会释放堆数据
3. 所有权可以移动来转移，一般在赋值和函数调用时候
4. 堆数据只能通过当前所有者访问，而不能通过之前的所有者访问

然而：每次移动都不能访问有点麻烦
引用：非拥有的指针，并不拥有指向的数据，也不改，只是打印or 访问看看
解引用：访问数据, *x, 

rust 必须避免同时发生别名和修改
指针能访问，但不能同时修改它！数据需要保证不能同时被别名和被修改
```rust
let mut v: Vec<i32> = vec![1, 2, 3];
let num: &i32 = &v[2];  // 别名
v.push(4);  // 可能分配新的堆来存储了，
println!("Third element is {}", *num); // error!
}
```
变量堆数据有3种权限
1. 可读，R
2. 可写，W
3. 拥有，O
使用权限，确保当数据存在别名时候，不能被修改。
默认是RO, mut 会添加RWO. 引用可以临时移除权限
![[Pasted image 20260525153948.png|521]]
这里先打印，后push 就可以。
注意：num, * num 不同，前者表示引用自己，后者表示引用操作的对象
权限是在位置上定义的，而不仅仅是变量。
位置指放在赋值左侧的任何东西，位置包括 变量，位置解引用* a, 数组访问
位置在不在使用时候，权限会丢失。只有在修改v 后再次使用num 才会有问题。
任何时候使用位置时候，rust 期望当前位置拥有我操作需要的那个权限（类似一致性协议，你要写数据需要有W 权限才行）
![[Pasted image 20260525160137.png|486]]&
&v[2] 要求有R 权限，push 要求有RW 权限
然而此时字母W 空心，v 不具备W 权限，被num 借用了！编译失败。
如果push 放在println 后面，就可以了，此时num 不再被使用了就释放了权限，这样v 重新获得了所有权限。

目前都是不可变引用，只读，允许别名但不能修改。
可变引用（unique references) 能在不转移数据所有权下临时提供对数据的可变访问。
使用&mut 来创建可变引用，&i32 是普通引用
```rust
let mut v: Vec<i32> = vec![1, 2, 3];  // v = RWO
let num: &mut i32 = &v[2];  // *num=RW,  &mut 可变引用
*num += 1;  // 暂时加1， [1,2,4]
println({*num})
println({v})
```

权限在引用生命周期结束后归还。