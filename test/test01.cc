#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// copy from src/binary_ops.cc
struct _Tensor {
    int dtype;
    uint64_t addr;
    int32_t dims;
    int64_t nelems;
    int64_t dim_size[8];

    std::string to_s() const {
        std::stringstream s;

        s << "[dtype=" << dtype
            << ",dims=" << dims
            << "[";
        for (int i = 0; i < dims; ++i)
            s << " " << dim_size[i];
        s  << " ],nelems=" << nelems
            << "]";
        return s.str();
    }
};

struct BinaryOpArgs {
    _Tensor in0;
    _Tensor in1;
    _Tensor out;
};

template<typename T> struct dypte_s {};
template<> struct dypte_s<float> { static const int type = 1; };

template <typename T>
_Tensor makeTensor(size_t dims, std::vector<size_t> const& dim_size)
{
    _Tensor t;

    t.dtype = dypte_s<T>::type;
    t.dims = dims;
    t.nelems = 1;
    for (int i = 0; i < dims; ++i) {
        t.dim_size[i] = dim_size[i];
        t.nelems *= dim_size[i];
    }

    t.addr = reinterpret_cast<uint64_t>(new T[t.nelems]);

    return t;
}

template<typename T>
class Tensor {
    public:
        Tensor(std::vector<size_t> const& shape) {
          shape_ = shape;
          t = makeTensor<T>(shape.size(), shape);
          stride_.resize(shape.size());
          size_t dim = t.dims;
          stride_[dim - 1] = 1;
          for (int i = dim - 2; i >= 0; --i) {
            stride_[i] = stride_[i + 1] * t.dim_size[i + 1];
          }
        }
        ~Tensor() { delete[] reinterpret_cast<T*>(t.addr); }
        std::vector<size_t> const& shape() const { return shape_; }
        T* data() { return reinterpret_cast<T*>(t.addr); }
        T const* data() const { return reinterpret_cast<T const*>(t.addr); }
        size_t nelems() const { return t.nelems; }
        size_t dims() const { return t.dims; }
        size_t dim_size(size_t i) const { return t.dim_size[i]; }
        size_t stride(size_t i) const { return stride_[i]; }

        _Tensor tensor() const { return t; }

    private:
        _Tensor t;
        std::vector<size_t> stride_;
        std::vector<size_t> shape_;
};

template<typename T>
bool checkTensor(Tensor<T> const& a, Tensor<T> const& b)
{
    if (a.nelems() != b.nelems())
        return false;

    for (size_t i = 0; i < a.nelems(); ++i)
        if (a.data()[i] != b.data()[i])
            return false;
    return true;
}

template<typename T>
void printTensor(Tensor<T> const& t, std::string fmt = " %8.3f")
{
    std::vector<size_t> s(t.dims() + 1);
    s[t.dims()] = 1;
    for (int i = t.dims() - 1; i >= 0; --i)
        s[i] = s[i + 1] * t.dim_size(i);

#if 0
    fprintf(stderr, "%d %d %d\n", t.dim_size(0), t.dim_size(1), t.dim_size(2));
    fprintf(stderr, "%d %d %d\n", s[0], s[1], s[2]);
#endif

    float const* p = t.data();
    size_t n = t.dim_size(t.dims() - 1); // innermost

    for (size_t i = 0; i < t.nelems(); ++i) {
        if (i % n == 0) {
            for (int j = 0; j < t.dims(); ++j) {
                fprintf(stderr, "%c", i % s[j] == 0 ? '[' : ' ');
            }
        }
        fprintf(stderr, fmt.c_str(), p[i]);
        if ((i + 1) % n == 0) {
            fprintf(stderr, " ");
            for (int j = 0; j < t.dims(); ++j) {
                if ((i + 1) % s[j] == 0) 
                    fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
    }
}

struct TestParam
{
    int verbose;
};

extern "C" {
    int op_Add(const void* args, size_t len);
    int op_Sub(const void* args, size_t len);
    int op_Mul(const void* args, size_t len);
}

template<typename T>
bool test_BinaryOp(TestParam const& param,
                   Tensor<T>& out, Tensor<T> const& in0, Tensor<T> const& in1, Tensor<T> const& exp,
                   int (*op)(const void* args, size_t len))
{
    BinaryOpArgs args;
    args.out = out.tensor();
    args.in0 = in0.tensor();
    args.in1 = in1.tensor();
    int ret = op(&args, sizeof(args));

    bool flag = false;
    if (ret == 0)
        flag = checkTensor(out, exp);

    if (param.verbose > 1 || (!flag && param.verbose > 0)) {
        fprintf(stderr, "in0 = \n");
        printTensor(in0);
        fprintf(stderr, "in1 = \n");
        printTensor(in1);
        fprintf(stderr, "in0 + in1 = \n");
        printTensor(out);
        fprintf(stderr, "expected = \n");
        printTensor(exp);
    }

    return flag;
}

template <typename T, typename F>
int ref_Binop(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z, F op,
        T* pX, T const* pY, T const* pZ, int dim)
{
  //fprintf(stderr, "%s: dim=%d X.stride[%d]=%d\n", __FUNCTION__, dim, dim, X.stride(dim));
  if (dim + 1 == X.dims()) {
    for (size_t i = 0; i < X.dim_size(dim); ++i) {
      T y = pY[i % Y.dim_size(dim)];
      T z = pZ[i % Z.dim_size(dim)];
      pX[i] = op(y, z);
      //fprintf(stderr, "%s: %8.3f = %8.3f op %8.3f\n", __FUNCTION__, pX[i], y, z);
    }
  } else {
    for (size_t i = 0; i < X.dim_size(dim); ++i) {
#if 0
      fprintf(stderr, "%s: dim=%d X.dim_size[%d]=%d i=%d %d %d\n",
              __FUNCTION__, dim, dim, X.dim_size(dim), i, Y.dim_size(dim), Y.stride(dim));
#endif
      T* pX0 = pX + i * X.stride(dim);
      T const* pY0 = pY + (i % Y.dim_size(dim)) * Y.stride(dim);
      T const* pZ0 = pZ + (i % Z.dim_size(dim)) * Z.stride(dim);
      ref_Binop(X, Y, Z, op, pX0, pY0, pZ0, dim + 1);
    }
  }
  return 0;
}

template <typename T, typename F>
int ref_Binop(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z, F op)
{
  return ref_Binop(X, Y, Z, op, X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_Add(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y + z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_Sub(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y - z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_Mul(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y * z; },
          X.data(), Y.data(), Z.data(), 0);
}


bool test_Add_01(TestParam const& param)
{
    Tensor<float> out({1, 5, 10});
    Tensor<float> in0({1, 5, 10});
    Tensor<float> in1({1, 1, 10});
    Tensor<float> exp({1, 5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t i = 0; i < 1; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            in1.data()[j] = j * 100;
        }
    }

    ref_Add(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, op_Add);
}

bool test_Add_02(TestParam const& param)
{
    Tensor<float> out({1, 5, 10});
    Tensor<float> in0({1, 5, 10});
    Tensor<float> in1({1, 5,  1});
    Tensor<float> exp({1, 5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            //exp.data()[i * 10 + j] = c + i * 100;
            ++c;
        }
    }

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 1; ++j) {
            in1.data()[i] = i * 100;
        }
    }

    ref_Add(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, op_Add);
}

bool test_Add_03(TestParam const& param)
{
    Tensor<float> out({2, 3, 10});
    Tensor<float> in0({2, 1, 10});
    Tensor<float> in1({1, 3,  1});
    Tensor<float> exp({2, 3, 10});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    int c = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            in0.data()[i * 10 + j] = i * 10 + j;
        }
    }

    for (size_t i = 0; i < 3; ++i) {
        in1.data()[i] = i * 100;
    }

    ref_Add(exp, in0, in1);
    return test_BinaryOp(param, out, in0, in1, exp, op_Add);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_04(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 16, 32, 32});
    Tensor<T> in0({8, 16, 16, 32, 32});
    Tensor<T> in1({1, 16, 16, 1, 1});
    Tensor<T> exp({8, 16, 16, 32, 32});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

bool test_Add_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Add<float>, op_Add);
}

bool test_Sub_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Sub<float>, op_Sub);
}

bool test_Mul_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Mul<float>, op_Mul);
}


struct Test
{
    std::string name;
    bool (*func)(TestParam const&);
};

extern "C" {
    int get_num_kernels();
}

int main(int argc, char* argv[])
{
    fprintf(stderr, "num_kernels=%d\n", get_num_kernels());

    Test tests[] = {
        "op_Add_01", test_Add_01,
        "op_Add_02", test_Add_02,
        "op_Add_03", test_Add_03,
        "op_Add_04", test_Add_04,
        "op_Sub_04", test_Sub_04,
        "op_Mul_04", test_Mul_04,
    };

    TestParam param;
    param.verbose = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            ++param.verbose;
        }
    }

    int ntests = sizeof(tests) / sizeof(Test);
    int ok = 0;
    for (size_t i = 0; i < ntests; ++i) {
        bool flag = tests[i].func(param);
        fprintf(stderr, "%-20s %s\n", tests[i].name.c_str(), flag ? "OK" : "NG");
        if (flag)
            ++ok;
    }
    fprintf(stderr, "%d tests failed\n", ntests - ok);
    return 0;
}
