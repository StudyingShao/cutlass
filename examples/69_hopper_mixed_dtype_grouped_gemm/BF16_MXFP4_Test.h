#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/numeric_conversion.h"

#include <cuda_bf16.h>

#include <bitset>

/*
ncu --clock-control none --kernel-id ::: \
-o 20250610_ncu_3090_FP4toBF16 -f --set full \
--import-source yes ./TEST_shao

*/



template<typename T>
void set_host_uint4(T *ptr_, int count) {

    float e2m1_values[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    for (unsigned int i = 0; i < count / 2; i++)
    {
        int offset = 2 + i;

        uint8_t low = (2 * offset) % 16; // 0~15
        uint8_t high = low + 1; // 0~15
        printf("%d low=%d (%f) high=%d (%f)\n", i, low, e2m1_values[low], high, e2m1_values[high]);
        uint8_t *ptr = reinterpret_cast<uint8_t *>(ptr_);
        ptr[i] = (high << 4) | low;
    }
}

__host__
__device__
void print_bits_32(uint32_t num) {
    for (int i = 31; i >= 0; i--) {       // 从最高位开始遍历[4,8](@ref)
        char c = (num & (1u << i)) ? '1' : '0';
        printf("%c", c);  // 按位判断并打印
        if (i % 4 == 0 && i != 0) {       // 每四位添加空格分隔[4](@ref)
            printf(" ");
        }
    }
    printf("\n");
}


__host__
__device__
void print_bits_16(uint16_t num) {
    for (int i = 15; i >= 0; i--) {       // 从最高位开始遍历[4,8](@ref)
        char c = (num & (1u << i)) ? '1' : '0';
        printf("%c", c);  // 按位判断并打印
        if (i % 4 == 0 && i != 0) {       // 每四位添加空格分隔[4](@ref)
            printf(" ");
        }
    }
    printf("\n");
}


__host__
__device__
void print_bits_8(uint8_t num) {
    for (int i = 7; i >= 0; i--) {       // 从最高位开始遍历[4,8](@ref)
        char c = (num & (1u << i)) ? '1' : '0';
        printf("%c", c);  // 按位判断并打印
        if (i % 4 == 0 && i != 0) {       // 每四位添加空格分隔[4](@ref)
            printf(" ");
        }
    }
    printf("\n");
}

void BF16_bit_test() {
    __nv_bfloat16 bf16_pos[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    __nv_bfloat16 bf16_neg[] = {-0, -0.5, -1, -1.5, -2, -3, -4, -6};

    for (size_t i = 0; i < 8; i++)
    {
        printf("%f\n", float(bf16_pos[i]));
        print_bits_16(*reinterpret_cast<uint16_t*>(&(bf16_pos[i])));
    }

    printf("------------------------------------------------------------------\n");

    for (size_t i = 0; i < 8; i++)
    {
        printf("%f\n", float(bf16_neg[i]));
        print_bits_16(*reinterpret_cast<uint16_t*>(&(bf16_neg[i])));
    }
}

void ue8m0_bit_test() {

    for (int i = 0; i < 256; i++)
    {
        uint8_t value_uint8 = i;
        cutlass::float_ue8m0_t value_ue8m0 = *reinterpret_cast<cutlass::float_ue8m0_t *>(&value_uint8);
        printf("%d %f\n", int(value_uint8), float(value_ue8m0));
        print_bits_8(*reinterpret_cast<uint16_t*>(&value_ue8m0));
    }
}


/*****************************lut prmt*********************************/
typedef uint32_t            __nv_fp4x8_storage_t;
typedef uint32_t            __nv_bf16x2_storage_t;
typedef uint64_t            __nv_fp8x8_storage_t;
typedef cutlass::uint128_t  __nv_bf16x8_storage_t;

inline __device__ unsigned
prmt(unsigned hi, unsigned lo, unsigned select_code)
{

    unsigned res = 0;

    asm volatile(
	"{\n"							\
	"prmt.b32 %0, %1, %2, %3;\n"				\
	"}\n"							\
	: "=r"(res) : "r"(lo) , "r"(hi), "r"(select_code));
    
    return res;
}

__device__ __inline__
__nv_fp8x4_storage_t
cvt_lut
(
    const unsigned index
)
{
    const __nv_fp8x4_storage_t h4b_lut = 0x4C484440U; //7654
    const __nv_fp8x4_storage_t l4b_lut = 0x3C383000U; //3210

    __nv_fp8x4_storage_t lut_res = prmt(h4b_lut, l4b_lut, index);

    return lut_res;
}

__device__ __inline__
__nv_fp8x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_fp8x8
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    __nv_fp8x8_storage_t fp8x8_raw = 0;
    
    __nv_fp8x4_storage_t hb_sign_fp8x4 = (fp4x8 & 0x80808080U);       // 80808080: 7654
    __nv_fp8x4_storage_t lb_sign_fp8x4 = (fp4x8 & 0x08080808U) << 4U; // 80808080: 3210

    __nv_fp8x4_storage_t h4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x7362U); //7362
    __nv_fp8x4_storage_t l4b_sign_fp8x4 = prmt(hb_sign_fp8x4, lb_sign_fp8x4, 0x5140U); //5140

    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);

    __nv_fp8x4_storage_t h4b_em_fp8x4 = cvt_lut(h4b_em_fp4x4);
    __nv_fp8x4_storage_t l4b_em_fp8x4 = cvt_lut(l4b_em_fp4x4);

    __nv_fp8x4_storage_t h4b_fp8x4_raw = h4b_sign_fp8x4 + h4b_em_fp8x4;
    __nv_fp8x4_storage_t l4b_fp8x4_raw = l4b_sign_fp8x4 + l4b_em_fp8x4;

    fp8x8_raw = (__nv_fp8x8_storage_t) h4b_fp8x4_raw;
    fp8x8_raw = (__nv_fp8x8_storage_t) (fp8x8_raw << 32U);
    fp8x8_raw = (__nv_fp8x8_storage_t) (fp8x8_raw | l4b_fp8x4_raw);
    
    return fp8x8_raw;
}

__device__ __inline__
__nv_fp8x4_storage_t
cvt_lut_bf16
(
    const unsigned index
)
{
    const __nv_fp8x4_storage_t h4b_lut = 0x03020100U; //7654
    const __nv_fp8x4_storage_t l4b_lut = 0xFFFEFC00U; //3210

    __nv_fp8x4_storage_t lut_res = prmt(h4b_lut, l4b_lut, index);

    return lut_res;
}


__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_bf16x8
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    __nv_bf16x8_storage_t bf16x8_raw = {0, 0};
    __nv_bf16x2_storage_t *bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t *>(&bf16x8_raw);
    
    unsigned zero_padding = 0x00000000U;

    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);
    
    __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_bf16(h4b_em_fp4x4); // 7654
    __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_bf16(l4b_em_fp4x4); // 3210

    bf16x2_raw[0] = prmt(zero_padding, l4b_2to9_bits, 0x1707U) >> 2U; // 1 0
    bf16x2_raw[1] = prmt(zero_padding, l4b_2to9_bits, 0x3727U) >> 2U; // 3 2
    bf16x2_raw[2] = prmt(h4b_2to9_bits, zero_padding, 0x5040U) >> 2U; // 5 4
    bf16x2_raw[3] = prmt(h4b_2to9_bits, zero_padding, 0x7060U) >> 2U; // 7 6

    __nv_bf16x2_storage_t bf16x2_0to1_bits;

    __nv_fp8x4_storage_t h_fp8x2_0to1_bits = (fp4x8 & 0x0000C0C0U);       // 3 1
    __nv_fp8x4_storage_t l_fp8x2_0to1_bits = (fp4x8 & 0x00000C0CU) << 4U; // 2 0

    bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x4707U); // 1 0
    bf16x2_raw[0] = bf16x2_raw[0] | bf16x2_0to1_bits;
    bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x5717U); // 3 2
    bf16x2_raw[1] = bf16x2_raw[1] | bf16x2_0to1_bits;

    h_fp8x2_0to1_bits = (fp4x8 & 0xC0C00000U);       // 7 5
    l_fp8x2_0to1_bits = (fp4x8 & 0x0C0C0000U) << 4U; // 6 4

    bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x6020U); // 5 4
    bf16x2_raw[2] = bf16x2_raw[2] | bf16x2_0to1_bits;
    bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x7030U); // 7 6
    bf16x2_raw[3] = bf16x2_raw[3] | bf16x2_0to1_bits;

    // print_bits_16(*reinterpret_cast<uint16_t *>(&bf16x2_raw[0]));
    // print_bits_16(*(reinterpret_cast<uint16_t *>(&bf16x2_raw[0]) + 1));
    // print_bits_16(*reinterpret_cast<uint16_t *>(&bf16x2_raw[1]));
    // print_bits_16(*(reinterpret_cast<uint16_t *>(&bf16x2_raw[1]) + 1));
    // print_bits_16(*reinterpret_cast<uint16_t *>(&bf16x2_raw[2]));
    // print_bits_16(*(reinterpret_cast<uint16_t *>(&bf16x2_raw[2]) + 1));
    // print_bits_16(*reinterpret_cast<uint16_t *>(&bf16x2_raw[3]));
    // print_bits_16(*(reinterpret_cast<uint16_t *>(&bf16x2_raw[3]) + 1));

    return bf16x8_raw;
}


__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    // interleaved version
    // input fp4x8: 7531 6420
    // output bf16x8: 7654 3210

    __nv_bf16x8_storage_t bf16x8_raw = {0, 0};
    __nv_bf16x2_storage_t *bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t *>(&bf16x8_raw);

    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);
    
    __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_bf16(h4b_em_fp4x4); // 7531
    __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_bf16(l4b_em_fp4x4); // 6420
    
    __nv_fp8x4_storage_t h4b_2to9_bits_pad0 = h4b_2to9_bits & 0x00FFFFFF; // [0]531

    bf16x2_raw[0] = prmt(h4b_2to9_bits_pad0, l4b_2to9_bits, 0x7470U) << 6U; // 1 0
    bf16x2_raw[1] = prmt(h4b_2to9_bits_pad0, l4b_2to9_bits, 0x7571U) << 6U; // 3 2

    h4b_2to9_bits_pad0 = h4b_2to9_bits & 0xFFFFFF00; // 753[0]

    bf16x2_raw[2] = prmt(h4b_2to9_bits_pad0, l4b_2to9_bits, 0x4642U) << 6U; // 5 4
    bf16x2_raw[3] = prmt(h4b_2to9_bits_pad0, l4b_2to9_bits, 0x4743U) << 6U; // 7 6

    __nv_fp4x8_storage_t fp4x8_ = fp4x8;

    // bf16x2_raw[3] = (fp4x8_ & 0xC000C000U) | (bf16x2_raw[3] & 0x3FFF3FFFU);
    // fp4x8_ = fp4x8_ << 4U;
    // bf16x2_raw[2] = (fp4x8_ & 0xC000C000U) | (bf16x2_raw[2] & 0x3FFF3FFFU);
    // fp4x8_ = fp4x8_ << 4U;
    // bf16x2_raw[1] = (fp4x8_ & 0xC000C000U) | (bf16x2_raw[1] & 0x3FFF3FFFU);
    // fp4x8_ = fp4x8_ << 4U;
    // bf16x2_raw[0] = (fp4x8_ & 0xC000C000U) | (bf16x2_raw[0] & 0x3FFF3FFFU);

    bf16x2_raw[3] = (fp4x8_ & 0xC000C000U) | bf16x2_raw[3];
    fp4x8_ = fp4x8_ << 4U;
    bf16x2_raw[2] = (fp4x8_ & 0xC000C000U) | bf16x2_raw[2];
    fp4x8_ = fp4x8_ << 4U;
    bf16x2_raw[1] = (fp4x8_ & 0xC000C000U) | bf16x2_raw[1];
    fp4x8_ = fp4x8_ << 4U;
    bf16x2_raw[0] = (fp4x8_ & 0xC000C000U) | bf16x2_raw[0];

    // __nv_bf16x2_storage_t bf16x2_0to1_bits;

    // __nv_fp8x4_storage_t h_fp8x2_0to1_bits = (fp4x8 & 0x0000C0C0U);       // 3 1
    // __nv_fp8x4_storage_t l_fp8x2_0to1_bits = (fp4x8 & 0x00000C0CU) << 4U; // 2 0

    // bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x4707U); // 1 0
    // bf16x2_raw[0] = bf16x2_raw[0] | bf16x2_0to1_bits;
    // bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x5717U); // 3 2
    // bf16x2_raw[1] = bf16x2_raw[1] | bf16x2_0to1_bits;

    // h_fp8x2_0to1_bits = (fp4x8 & 0xC0C00000U);       // 7 5
    // l_fp8x2_0to1_bits = (fp4x8 & 0x0C0C0000U) << 4U; // 6 4

    // bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x6020U); // 5 4
    // bf16x2_raw[2] = bf16x2_raw[2] | bf16x2_0to1_bits;
    // bf16x2_0to1_bits = prmt(h_fp8x2_0to1_bits, l_fp8x2_0to1_bits, 0x7030U); // 7 6
    // bf16x2_raw[3] = bf16x2_raw[3] | bf16x2_0to1_bits;

    return bf16x8_raw;
}

__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved_v2
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    // interleaved version
    // input fp4x8: 7362 5140
    // output bf16x8: 7654 3210

    __nv_bf16x8_storage_t bf16x8_raw;
    __nv_bf16x2_storage_t *bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t *>(&bf16x8_raw);

    __nv_fp8x4_storage_t h_fp8x4_0to1_bits = (fp4x8 & 0xC0C0C0C0U) >> 6; // 7654
    __nv_fp8x4_storage_t l_fp8x4_0to1_bits = (fp4x8 & 0x0C0C0C0CU) >> 2; // 3210
    
    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);

    __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_bf16(h4b_em_fp4x4); // 7362
    __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_bf16(l4b_em_fp4x4); // 5140

    bf16x2_raw[0] = prmt(l_fp8x4_0to1_bits, l4b_2to9_bits, 0x5240U) << 6U; // 1 0
    bf16x2_raw[1] = prmt(l_fp8x4_0to1_bits, h4b_2to9_bits, 0x7260U) << 6U; // 3 2

    bf16x2_raw[2] = prmt(h_fp8x4_0to1_bits, l4b_2to9_bits, 0x5341U) << 6U; // 5 4
    bf16x2_raw[3] = prmt(h_fp8x4_0to1_bits, h4b_2to9_bits, 0x7361U) << 6U; // 7 6

    return bf16x8_raw;
}

__device__ __inline__
__nv_bf16x8_storage_t
psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved_v3
(
    const __nv_fp4x8_storage_t fp4x8
)
{
    // interleaved version
    // input fp4x8: 7564 3120
    // output bf16x8: 7654 3210

    __nv_bf16x8_storage_t bf16x8_raw;
    __nv_bf16x2_storage_t *bf16x2_raw = reinterpret_cast<__nv_bf16x2_storage_t *>(&bf16x8_raw);

    __nv_fp8x4_storage_t h_fp8x4_0to1_bits = (fp4x8 & 0xC0C0C0C0U) >> 6; // 7632
    __nv_fp8x4_storage_t l_fp8x4_0to1_bits = (fp4x8 & 0x0C0C0C0CU) >> 2; // 5410
    
    unsigned h4b_em_fp4x4 = (fp4x8 & 0x77770000U) >> 16U;
    unsigned l4b_em_fp4x4 = (fp4x8 & 0x00007777U);

    __nv_fp8x4_storage_t h4b_2to9_bits = cvt_lut_bf16(h4b_em_fp4x4); // 7564
    __nv_fp8x4_storage_t l4b_2to9_bits = cvt_lut_bf16(l4b_em_fp4x4); // 3120

    bf16x2_raw[0] = prmt(l_fp8x4_0to1_bits, l4b_2to9_bits, 0x5240U) << 6U; // 1 0
    bf16x2_raw[1] = prmt(h_fp8x4_0to1_bits, l4b_2to9_bits, 0x5341U) << 6U; // 3 2

    bf16x2_raw[2] = prmt(l_fp8x4_0to1_bits, h4b_2to9_bits, 0x7260U) << 6U; // 5 4
    bf16x2_raw[3] = prmt(h_fp8x4_0to1_bits, h4b_2to9_bits, 0x7361U) << 6U; // 7 6

    return bf16x8_raw;
}


/*****************************lut prmt*********************************/



__global__ void fp4_to_fp8_kernel(cutlass::float_e2m1_t *fp4_ptr, cutlass::float_e4m3_t *fp8_ptr)
{
    __nv_fp4x8_storage_t *fp4x8_ptr = reinterpret_cast<__nv_fp4x8_storage_t *>(fp4_ptr);
    __nv_fp8x8_storage_t *fp8x8_ptr = reinterpret_cast<__nv_fp8x8_storage_t *>(fp8_ptr);

    *fp8x8_ptr = psx_cvt_lut_prmt_fp4x8_to_fp8x8(*fp4x8_ptr);
}

__global__ void fp4_to_bf16_kernel_fp4tobf16_lut(cutlass::float_e2m1_t *fp4_ptr, cutlass::bfloat16_t *bf16_ptr)
{    
    __nv_fp4x8_storage_t *fp4x8_ptr = reinterpret_cast<__nv_fp4x8_storage_t *>(fp4_ptr);
    __nv_bf16x8_storage_t *bf16x8_ptr = reinterpret_cast<__nv_bf16x8_storage_t *>(bf16_ptr);

    *bf16x8_ptr = psx_cvt_lut_prmt_fp4x8_to_bf16x8(*fp4x8_ptr);
}

__global__ void fp4_to_bf16_kernel_fp4tobf16_lut_interleaved(cutlass::float_e2m1_t *fp4_ptr, cutlass::bfloat16_t *bf16_ptr)
{    
    __nv_fp4x8_storage_t *fp4x8_ptr = reinterpret_cast<__nv_fp4x8_storage_t *>(fp4_ptr);
    __nv_bf16x8_storage_t *bf16x8_ptr = reinterpret_cast<__nv_bf16x8_storage_t *>(bf16_ptr);

    // *bf16x8_ptr = psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved(*fp4x8_ptr);
    // *bf16x8_ptr = psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved_v2(*fp4x8_ptr);
    *bf16x8_ptr = psx_cvt_lut_prmt_fp4x8_to_bf16x8_interleaved_v3(*fp4x8_ptr);
}

__global__ void fp4_to_bf16_kernel_fp4tofp8tobf16(cutlass::float_e2m1_t *fp4_ptr, cutlass::bfloat16_t *bf16_ptr)
{
    using FragmentFP8 = cutlass::Array<cutlass::float_e4m3_t, 8>;
    using FragmentBF16 = cutlass::Array<cutlass::bfloat16_t, 8>;
    
    FragmentFP8 fp8x8_frag;
    FragmentBF16 bf16x8_frag;
    
    __nv_fp4x8_storage_t *fp4x8_ptr = reinterpret_cast<__nv_fp4x8_storage_t *>(fp4_ptr);
    __nv_fp8x8_storage_t *fp8x8_ptr = reinterpret_cast<__nv_fp8x8_storage_t *>(&fp8x8_frag);
    
    *fp8x8_ptr = psx_cvt_lut_prmt_fp4x8_to_fp8x8(*fp4x8_ptr);

    cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e4m3_t, 8> converter;

    bf16x8_frag = converter(fp8x8_frag);

    *reinterpret_cast<float4 *>(bf16_ptr) = *reinterpret_cast<float4 *>(&bf16x8_frag);
}

__global__ void fp4_to_bf16_kernel_fp4tobf16_convert(cutlass::float_e2m1_t *fp4_ptr, cutlass::bfloat16_t *bf16_ptr)
{
    using FragmentFP4 = cutlass::Array<cutlass::float_e2m1_t, 8>;
    using FragmentBF16 = cutlass::Array<cutlass::bfloat16_t, 8>;
    
    FragmentFP4 fp4x8_frag;
    FragmentBF16 bf16x8_frag;
    
    fp4x8_frag = *reinterpret_cast<FragmentFP4 *>(fp4_ptr);
    
    cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e2m1_t, 8> converter;

    bf16x8_frag = converter(fp4x8_frag);

    *reinterpret_cast<float4 *>(bf16_ptr) = *reinterpret_cast<float4 *>(&bf16x8_frag);
}

__global__ void fp4_to_bf16_kernel_fp4tofp32tobf16_convert(cutlass::float_e2m1_t *fp4_ptr, cutlass::bfloat16_t *bf16_ptr)
{
    using FragmentFP4 = cutlass::Array<cutlass::float_e2m1_t, 8>;
    using FragmentFP32 = cutlass::Array<float, 8>;
    using FragmentBF16 = cutlass::Array<cutlass::bfloat16_t, 8>;
    
    FragmentFP4 fp4x8_frag;
    FragmentFP32 fp32x8_frag;
    FragmentBF16 bf16x8_frag;
    
    fp4x8_frag = *reinterpret_cast<FragmentFP4 *>(fp4_ptr);
    
    cutlass::NumericArrayConverter<float, cutlass::float_e2m1_t, 8> converter_fp4tofp32;
    cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, 8> converter_fp32tobf16;

    fp32x8_frag = converter_fp4tofp32(fp4x8_frag);
    bf16x8_frag = converter_fp32tobf16(fp32x8_frag);

    *reinterpret_cast<float4 *>(bf16_ptr) = *reinterpret_cast<float4 *>(&bf16x8_frag);
}


__global__ void prmt_test()
{
    unsigned a = 0x76543210;
    unsigned b = 0xFEDCBA98;
    unsigned c;
    unsigned d;
    printf("a = %08X\n", a);
    printf("b = %08X\n", b);

    c = 0x7610;
    d = prmt(b, a, c);
    printf("c = %04X\n", c);
    printf("d = %08X\n", d);

    c = 0xFE98;
    d = prmt(b, a, c);
    printf("c = %04X\n", c);
    printf("d = %08X\n", d);

    c = 0xFEC8;
    d = prmt(b, a, c);
    printf("c = %04X\n", c);
    printf("d = %08X\n", d);


}

__global__ void interleave_fp4_kernel(
    cutlass::float_e2m1_t *fp4_ptr, 
    cutlass::float_e2m1_t *fp4_interleaved_ptr, 
    const int rows, 
    const int cols
) {
    uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(fp4_ptr);
    uint16_t *uint16_interleaved_ptr = reinterpret_cast<uint16_t *>(fp4_interleaved_ptr);

    for (int row_id = blockIdx.x; row_id < rows; row_id += gridDim.x)
    {
        for (int col_id = threadIdx.x; col_id < cols / 4; col_id += blockDim.x)
        {
            int index = row_id * cols / 4 + col_id; // column-major
            
            uint16_t fp4x4 = uint16_ptr[index];
            uint16_t fp4_1 = (fp4x4 & 0x00F0) << 4;
            uint16_t fp4_2 = (fp4x4 & 0x0F00) >> 4;
            fp4x4 = (fp4x4 & 0xF00F) | fp4_1 | fp4_2;

            uint16_interleaved_ptr[index] = fp4x4;
        }
    }
}

void interleave_fp4(
    cutlass::float_e2m1_t *fp4_ptr, 
    cutlass::float_e2m1_t *fp4_interleaved_ptr, 
    const int rows, 
    const int cols
) {
    // column-major input
    interleave_fp4_kernel<<<1024, 1024>>>(fp4_ptr, fp4_interleaved_ptr, rows, cols);
}

template<typename T>
__global__ void interleave_fp4_Hopper_kernel(
    T *fp4_ptr, 
    T *fp4_interleaved_ptr, 
    const int rows, 
    const int cols
) {
    uint8_t *uint8_ptr = reinterpret_cast<uint8_t *>(fp4_ptr);
    uint8_t *uint8_interleaved_ptr = reinterpret_cast<uint8_t *>(fp4_interleaved_ptr);

    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int col_id = threadIdx.x; col_id < cols / 2; col_id += blockDim.x)
        {
            int row_id = block_id / 8 * 16 + block_id % 8;

            int index_a = row_id * cols / 2 + col_id; // row-major
            int index_b = (row_id + 8) * cols / 2 + col_id; // row-major
            
            uint8_t fp4x2_a = uint8_ptr[index_a];
            uint8_t fp4x2_b = uint8_ptr[index_b];

            uint8_t fp4_temp_a = (fp4x2_a & 0xF0U) >> 4;
            uint8_t fp4_temp_b = (fp4x2_b & 0x0FU) << 4;

            fp4x2_a = (fp4x2_a & 0x0FU) | fp4_temp_b;
            fp4x2_b = (fp4x2_b & 0xF0U) | fp4_temp_a;

            uint8_interleaved_ptr[index_a] = fp4x2_a;
            uint8_interleaved_ptr[index_b] = fp4x2_b;
        }
    }
}

template<typename T>
__global__ void interleave_w4a16_Hopper_kernel(
    T *fp4_ptr, 
    T *fp4_interleaved_ptr, 
    const int rows, 
    const int cols
) {
    uint8_t *uint8_ptr = reinterpret_cast<uint8_t *>(fp4_ptr);
    uint8_t *uint8_interleaved_ptr = reinterpret_cast<uint8_t *>(fp4_interleaved_ptr);
    // int64_t *uint8_ptr = reinterpret_cast<int64_t *>(fp4_ptr);
    // int64_t *uint8_interleaved_ptr = reinterpret_cast<int64_t *>(fp4_interleaved_ptr);

    for (int row_id = blockIdx.x; row_id < rows; row_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;
            int interleaved_lane_id = ((lane_id / 4) % 2) * 16 + (lane_id % 4) * 4 + lane_id / 8;

            int src_id = row_id * cols / 2 + partition_id * 32 + lane_id; // row-major
            int dst_id = row_id * cols / 2 + partition_id * 32 + interleaved_lane_id; // row-major

            uint8_interleaved_ptr[dst_id] = uint8_ptr[src_id];
        }
    }
}



template<typename T>
__global__ void interleave_w4a16_Hopper_kernel_combine(
    T *ptr_4b, 
    T *ptr_4b_interleaved, 
    const int rows, 
    const int cols
) {
    uint8_t *uint8_ptr = reinterpret_cast<uint8_t *>(ptr_4b);
    uint8_t *uint8_interleaved_ptr = reinterpret_cast<uint8_t *>(ptr_4b_interleaved);

    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;
            int interleaved_lane_id = ((lane_id / 4) % 2) * 16 + (lane_id % 4) * 4 + lane_id / 8;

            int col_id = partition_id * 32 + lane_id;
            int row_id = block_id / 8 * 16 + block_id % 8;

            int index_a = row_id * cols / 2 + col_id;
            int index_b = (row_id + 8) * cols / 2 + col_id;
            
            uint8_t fp4x2_a = uint8_ptr[index_a];
            uint8_t fp4x2_b = uint8_ptr[index_b];

            uint8_t fp4_temp_a = (fp4x2_a & 0xF0U) >> 4;
            uint8_t fp4_temp_b = (fp4x2_b & 0x0FU) << 4;

            fp4x2_a = (fp4x2_a & 0x0FU) | fp4_temp_b;
            fp4x2_b = (fp4x2_b & 0xF0U) | fp4_temp_a;

            int dst_col_id = partition_id * 32 + interleaved_lane_id;

            index_a = row_id * cols / 2 + dst_col_id;
            index_b = (row_id + 8) * cols / 2 + dst_col_id;

            uint8_interleaved_ptr[index_a] = fp4x2_a;
            uint8_interleaved_ptr[index_b] = fp4x2_b;
        }
    }
}


template<typename T>
__global__ void interleave_fp4xbf16_Hopper_kernel_combine_opt(
    T *ptr_4b, 
    T *ptr_4b_interleaved, 
    const int rows, 
    const int cols
) {
    // int64_t *uint8_ptr = reinterpret_cast<int64_t *>(ptr_4b);
    // int64_t *uint8_interleaved_ptr = reinterpret_cast<int64_t *>(ptr_4b_interleaved);
    uint8_t *uint8_ptr = reinterpret_cast<uint8_t *>(ptr_4b);
    uint8_t *uint8_interleaved_ptr = reinterpret_cast<uint8_t *>(ptr_4b_interleaved);

    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;
            int row_id = block_id / 8 * 16 + block_id % 8;
            
            int mma_id = lane_id / 8;
            int dst_row_id = row_id + (mma_id % 2) * 8;

            int interleaved_lane_id = lane_id / 16 * 16 + (lane_id % 4) * 4 + (lane_id % 8) / 4 * 2;
            
            int col_id = partition_id * 32 + lane_id;
            int dst_col_id = partition_id * 32 + interleaved_lane_id;

            int index_a = row_id * cols / 2 + col_id;
            int index_b = (row_id + 8) * cols / 2 + col_id;
            
            uint8_t fp4x2_a = uint8_ptr[index_a];
            uint8_t fp4x2_b = uint8_ptr[index_b];

            uint8_t fp4_temp_a = (fp4x2_a & 0xF0U) >> 4;
            uint8_t fp4_temp_b = (fp4x2_b & 0x0FU) << 4;

            fp4x2_a = (fp4x2_a & 0x0FU) | fp4_temp_b;
            fp4x2_b = (fp4x2_b & 0xF0U) | fp4_temp_a;

            int dst_id = dst_row_id * cols / 2 + dst_col_id;

            uint8_interleaved_ptr[dst_id] = fp4x2_a;
            uint8_interleaved_ptr[dst_id + 1] = fp4x2_b;
        }
    }
}


template<typename T>
__global__ void interleave_int4xfp8_Hopper_kernel(
    T *ptr_4b, 
    T *ptr_4b_interleaved, 
    const int rows, 
    const int cols
) {
    uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(ptr_4b);
    uint16_t *uint16_interleaved_ptr = reinterpret_cast<uint16_t *>(ptr_4b_interleaved);

    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;
            
            int row_id = block_id / 8 * 16 + block_id % 8;
            int dst_row_id = row_id + (lane_id % 8) / 4 * 8;

            int mma_id = lane_id / 8;
            int interleaved_lane_id = mma_id * 8 + lane_id % 4 * 2;
                        
            int col_id = partition_id * 16 + lane_id;
            int dst_col_id = partition_id * 16 + interleaved_lane_id;

            int src_id_a = row_id * cols / 4 + col_id;
            int src_id_b = (row_id + 8) * cols / 4 + col_id;
            
            uint16_t fp4x2_a = uint16_ptr[src_id_a];
            uint16_t fp4x2_b = uint16_ptr[src_id_b];

            int dst_id = dst_row_id * cols / 4 + dst_col_id;

            uint16_interleaved_ptr[dst_id] = fp4x2_a;
            uint16_interleaved_ptr[dst_id + 1] = fp4x2_b;
        }
    }
}


template<typename T>
void interleave_fp4_Hopper(
    T *fp4_ptr, 
    T *fp4_interleaved_ptr,
    const int rows, 
    const int cols
) {
    // row-major input
    interleave_fp4_Hopper_kernel<<<1024, 1024>>>(fp4_ptr, fp4_interleaved_ptr, rows, cols);

    // // row-major input
    // dim3 block(32, 32);
    // interleave_w4a16_Hopper_kernel<<<1024, block>>>(fp4_interleaved_ptr, fp4_interleaved_ptr_, rows, cols);
}


template<typename T>
void interleave_fp4xbf16_Hopper(
    T *fp4_ptr, 
    T *fp4_interleaved_ptr,
    const int rows, 
    const int cols
) {
    // row-major input
    dim3 block(32, 32);
    // interleave_w4a16_Hopper_kernel_combine<<<1024, block>>>(fp4_ptr, fp4_interleaved_ptr, rows, cols);
    interleave_fp4xbf16_Hopper_kernel_combine_opt<<<1024, block>>>(fp4_ptr, fp4_interleaved_ptr, rows, cols);
}


template<typename T>
void interleave_int4xfp8_Hopper(
    T *int4_ptr, 
    T *int4_interleaved_ptr,
    const int rows, 
    const int cols
) {
    // row-major input
    dim3 block(16, 32);
    interleave_int4xfp8_Hopper_kernel<<<1024, block>>>(int4_ptr, int4_interleaved_ptr, rows, cols);
}


void interleave_w4a16_Hopper_test()
{
    const int rows = 32;
    const int cols = 128;

    using T = cutlass::uint4b_t;

    T a[rows * cols];
    T a_interleaved[rows * cols];

    T *d_a;
    T *d_a_interleaved;

    cudaMalloc((void **)&d_a, rows * cols * sizeof(T));
    cudaMalloc((void **)&d_a_interleaved, rows * cols * sizeof(T));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols / 2; j++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(a);
            ptr[i * cols / 2 + j] = uint8_t((2 * j) % 16) | (uint8_t((2 * j + 1) % 16) << 4);
        }
    }
    
    cudaMemcpy(d_a, a, rows * cols * sizeof(T), cudaMemcpyHostToDevice);

    // interleave_fp4_Hopper(d_a, d_a_interleaved, rows, cols);
    // interleave_fp4xbf16_Hopper(d_a, d_a_interleaved, rows, cols);
    interleave_int4xfp8_Hopper(d_a, d_a_interleaved, rows, cols);

    cudaMemcpy(a_interleaved, d_a_interleaved, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < rows; i++) {
        printf("row %d:  ", i);
        for (int j = 0; j < cols / 2; j++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(a);
            printf("(%d): %d ", 2 * j, ptr[i * cols / 2 + j] & 0x0F);
            printf("(%d): %d ", 2 * j + 1, (ptr[i * cols / 2 + j] & 0xF0) >> 4);
        }
        printf("\n");
    }

    for (int i = 0; i < rows; i++) {
        printf("row %d:  ", i);
        for (int j = 0; j < cols / 2; j++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(a_interleaved);
            printf("(%d): %d ", 2 * j, ptr[i * cols / 2 + j] & 0x0F);
            printf("(%d): %d ", 2 * j + 1, (ptr[i * cols / 2 + j] & 0xF0) >> 4);
        }
        printf("\n");
    }
}


void interleave_fp4_Hopper_test()
{

    const int rows = 16;
    const int cols = 128;

    const int num = rows * cols;

    cutlass::float_e2m1_t fp4_a[num];
    cutlass::bfloat16_t   bf16_a[num];
    cutlass::bfloat16_t   bf16_a_interleaved[num];

    cutlass::float_e2m1_t *d_fp4_a;
    cutlass::float_e2m1_t *d_fp4_a_interleaved;
    cutlass::bfloat16_t   *d_bf16_a;
    cutlass::bfloat16_t   *d_bf16_a_interleaved;

    cudaMalloc((void **)&d_fp4_a, num * sizeof(cutlass::float_e2m1_t));
    cudaMalloc((void **)&d_fp4_a_interleaved, num * sizeof(cutlass::float_e2m1_t));
    cudaMalloc((void **)&d_bf16_a, num * sizeof(cutlass::bfloat16_t));
    cudaMalloc((void **)&d_bf16_a_interleaved, num * sizeof(cutlass::bfloat16_t));

    set_host_uint4(fp4_a, num);

    cudaMemcpy(d_fp4_a, fp4_a, num * sizeof(cutlass::float_e2m1_t), cudaMemcpyHostToDevice);

    interleave_fp4xbf16_Hopper<cutlass::float_e2m1_t>(d_fp4_a, d_fp4_a_interleaved, 4, num / 4);

    printf("sizeof(cutlass::float_e2m1_t) = %lu\n", sizeof(cutlass::float_e2m1_t));
    
    for (int i=0; i < num / 8; i++) {
        fp4_to_bf16_kernel_fp4tobf16_lut<<<1, 1>>>(
            d_fp4_a + 4 * i, d_bf16_a + 8 * i
        );
        fp4_to_bf16_kernel_fp4tobf16_lut<<<1, 1>>>(
            d_fp4_a_interleaved + 4 * i, d_bf16_a_interleaved + 8 * i
        );
    }

    cudaMemcpy(bf16_a, d_bf16_a, num * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bf16_a_interleaved, d_bf16_a_interleaved, num * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);

    for (int i=0; i < num; i++)
        printf("%d %d bf16  %f  %f\n", i / 16, i % 16, float(bf16_a[i]), float(bf16_a_interleaved[i]));

}

void MXFP4_test(){

    // prmt_test<<<1, 1>>>();

    cutlass::float_e2m1_t fp4_a[8];
    cutlass::float_e4m3_t fp8_a[8];

    cutlass::bfloat16_t   bf16_a[8];
    cutlass::bfloat16_t   bf16_a_interleaved[8];
    cutlass::bfloat16_t   bf16_b[8];
    cutlass::bfloat16_t   bf16_c[8];
    cutlass::bfloat16_t   bf16_d[8];

    cutlass::float_e2m1_t *d_fp4_a;
    cutlass::float_e2m1_t *d_fp4_a_interleaved;
    cutlass::float_e4m3_t *d_fp8_a;

    cutlass::bfloat16_t   *d_bf16_a;
    cutlass::bfloat16_t   *d_bf16_a_interleaved;
    cutlass::bfloat16_t   *d_bf16_b;
    cutlass::bfloat16_t   *d_bf16_c;
    cutlass::bfloat16_t   *d_bf16_d;

    cudaMalloc((void **)&d_fp4_a, 8 * sizeof(cutlass::float_e2m1_t));
    cudaMalloc((void **)&d_fp4_a_interleaved, 8 * sizeof(cutlass::float_e2m1_t));
    cudaMalloc((void **)&d_fp8_a, 8 * sizeof(cutlass::float_e4m3_t));

    cudaMalloc((void **)&d_bf16_a, 8 * sizeof(cutlass::bfloat16_t));
    cudaMalloc((void **)&d_bf16_a_interleaved, 8 * sizeof(cutlass::bfloat16_t));
    cudaMalloc((void **)&d_bf16_b, 8 * sizeof(cutlass::bfloat16_t));
    cudaMalloc((void **)&d_bf16_c, 8 * sizeof(cutlass::bfloat16_t));
    cudaMalloc((void **)&d_bf16_d, 8 * sizeof(cutlass::bfloat16_t));

    set_host_uint4(fp4_a, 8);

    cudaMemcpy(d_fp4_a, fp4_a, 8 * sizeof(cutlass::float_e2m1_t), cudaMemcpyHostToDevice);

    interleave_fp4(d_fp4_a, d_fp4_a_interleaved, 1, 8);

    fp4_to_fp8_kernel<<<1, 1>>>(d_fp4_a, d_fp8_a);
    cudaDeviceSynchronize();
    
    cudaMemcpy(fp8_a, d_fp8_a, 8 * sizeof(cutlass::float_e4m3_t), cudaMemcpyDeviceToHost);

    for (int i=0; i < 8; i++)
        printf("fp8  %f\n", float(fp8_a[i]));
    
    fp4_to_bf16_kernel_fp4tobf16_lut<<<1, 1>>>(d_fp4_a, d_bf16_a);
    fp4_to_bf16_kernel_fp4tobf16_lut_interleaved<<<1, 1>>>(d_fp4_a_interleaved, d_bf16_a_interleaved);
    fp4_to_bf16_kernel_fp4tofp8tobf16<<<1, 1>>>(d_fp4_a, d_bf16_b);
    fp4_to_bf16_kernel_fp4tobf16_convert<<<1, 1>>>(d_fp4_a, d_bf16_c);
    fp4_to_bf16_kernel_fp4tofp32tobf16_convert<<<1, 1>>>(d_fp4_a, d_bf16_d);

    cudaDeviceSynchronize();
    
    cudaMemcpy(bf16_a, d_bf16_a, 8 * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bf16_a_interleaved, d_bf16_a_interleaved, 8 * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bf16_b, d_bf16_b, 8 * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bf16_c, d_bf16_c, 8 * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bf16_d, d_bf16_d, 8 * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost);

    for (int i=0; i < 8; i++)
        printf("%d  bf16  %f  %f %f  %f  %f\n", i, float(bf16_a[i]), float(bf16_a_interleaved[i]), float(bf16_b[i]), float(bf16_c[i]), float(bf16_d[i]));


    // printf("sizeof(__nv_fp4x8_storage_t) = %ld\n", sizeof(__nv_fp4x8_storage_t));
    // printf("sizeof(__nv_fp8x8_storage_t) = %ld\n", sizeof(__nv_fp8x8_storage_t));
    // printf("sizeof(__nv_bf16x8_storage_t) = %ld\n", sizeof(__nv_bf16x8_storage_t));    

}


void printFloatBits(float f) {
    // 1. 确保我们有一个 32 位的无符号整数类型来操作
    uint32_t bits;
    
    // 2. 安全地将 float 的位表示复制到 uint32_t 中
    //    使用 memcpy 避免类型双关（type-punning）问题，这是符合 C++ 标准的做法
    std::memcpy(&bits, &f, sizeof(float));
    
    // 3. 使用 std::bitset 来方便地获取和操作二进制位
    std::bitset<32> bitset(bits);
    
    // 4. 格式化输出：
    std::cout << "Float value: " << f << std::endl;
    std::cout << "Binary representation (Sign-Exponent-Mantissa):" << std::endl;
    
    // 提取并输出符号位 (第31位, 最高位)
    std::cout << "S:  " << bitset[31] << " | ";
    
    // 提取并输出指数位 (第30位到第23位)
    std::cout << "E:  ";
    for (int i = 30; i >= 23; --i) {
        std::cout << bitset[i];
    }
    std::cout << " | ";
    
    // 提取并输出尾数位 (第22位到第0位)
    std::cout << "M:  ";
    for (int i = 22; i >= 0; --i) {
        std::cout << bitset[i];
    }
    std::cout << std::endl;
    
    // 5. (可选) 输出连续的二进制字符串（如果需要原始位序列）
    // std::cout << "Raw bits: " << bitset << std::endl;
}

template <class ScaleType>
float scale_convertor(ScaleType scale) {

  if constexpr (cute::is_same_v<ScaleType, float>) {
    return scale;
  } 
  else if constexpr (cute::is_same_v<ScaleType, cutlass::half_t> || cute::is_same_v<ScaleType, cutlass::bfloat16_t>) {
    return static_cast<float>(scale);
  }
  else if constexpr (cute::is_same_v<ScaleType, cutlass::float_ue8m0_t>) {
    uint32_t temp = 0;
    temp = (temp | *reinterpret_cast<uint8_t*>(&scale)) << 23;
    return *reinterpret_cast<float*>(&temp);
  } 
  else {
    static_assert(cutlass::detail::dependent_false<ScaleType>, "Unsupported scale type");
  }
}

void UE8M0_FP32_test() {
    ue8m0_bit_test();

    uint8_t uint8_a[8] = {124, 125, 126, 127, 128, 129, 130, 131};
    cutlass::float_ue8m0_t ue8m0_a[8];
    float fp32_a[8];

    for (int i = 0; i < 8; i++) {
        ue8m0_a[i] = *reinterpret_cast<cutlass::float_ue8m0_t*>(&uint8_a[i]);
        // uint32_t temp = 0;
        // temp = (temp | *reinterpret_cast<uint8_t*>(&ue8m0_a[i])) << 23;
        // fp32_a[i] = *reinterpret_cast<float*>(&temp);
        fp32_a[i] = scale_convertor(ue8m0_a[i]);
    }

    printf("uint8 to ue8m0:\n");
    for (int i = 0; i < 8; i++) {
        printf("%d %f %f\n", int(uint8_a[i]), float(ue8m0_a[i]), fp32_a[i]);
        // printFloatBits(fp32_a[i]);
    }
}

