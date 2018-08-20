## build

    % mkdir build
    % cd build
    % cmake3 ..
    % make

## 実行

samplesリポジトリをご参考に

## カーネルの作成

TF側にVEを呼び出すためのカーネルと，VE側で実際に計算するカーネルを実装する．

Conv2D用にサンプル実装してあるのでご参考に．
- TF側: tensorflow/tensorflow/core/kernels/conv_ops.cc
- VE側: libvetfkernel/src/conv2d.cc

### TF側コード

VEで実行したいカーネルを実装する．

OpKernel::Compute()で，VEに渡したい引数をシリアライズして，VEDeviceContext::Compute()を呼び出すように実装する．

Computeの先頭引数はカーネル名．カーネル名でVE側で呼ばれるカーネルを識別する．


```
void Compute(OpKernelContext* context) override {
    // serialize arguments
    VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
    Status s = vectx->Compute("Conv2D", (void*)&p, sizeof(p));
}
```

カーネルを登録する

```
REGISTER_KERNEL_BUILDER(
      Name("Conv2D").Device(DEVICE_VE).TypeConstraint<float>("T"),
      Conv2DOp<VEDevice, float>);
```

### VE側コード(libvetfkernelで実装）

- kernel.hをinclude
- カーネルの演算をする関数を実装する
    - int *func*(const void* args, size_t len)
    - funcは自分の関数名に置き換える．argsはシリアライズした引数へのポインタ．lenは引数のサイズ．
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
