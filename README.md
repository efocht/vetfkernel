## build

    % mkdir build
    % cd build
    % cmake3 ..
    % make


## カーネルの作成

- TF内でkernelを実装．引数をシリアライズしてVEDeviceContext::Compute()を呼び出す
    - conv_ops.ccに実装例あり
- libfetfkernelで，VE用コードを実装．
    - kernel.hをinclude
    - 以下の関数が入口．conv2dは自分の関数名に置き換える．argsはシリアライズした引数へのポインタ．lenは引数のサイズ．
- REGISTER_KERNEL(カーネル名，関数名)でカーネルを登録．C++の場合は関数名に注意．extern "C" しておくのが良い


```
#include "kernel.h"

REGISTER_KENREL("Conv2D", "conv2d")

extern "C" {
  int conv2d(const void* args, size_t len);
}

int conv2d(const void* args, size_t len) {
    ....
}
```
