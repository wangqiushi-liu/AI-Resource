#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * 卷积操作实现
 * 支持基本的2D卷积操作，用于深度学习中的卷积层
 */

// 卷积操作函数
void conv2d(const float* input,           // 输入数据 (height x width x channels)
            int input_height,             // 输入高度
            int input_width,              // 输入宽度
            int input_channels,           // 输入通道数
            const float* kernel,          // 卷积核 (kernel_height x kernel_width x input_channels x output_channels)
            int kernel_height,            // 卷积核高度
            int kernel_width,             // 卷积核宽度
            int output_channels,          // 输出通道数
            int stride,                    // 步长
            int padding,                   // 填充大小
            float* output) {              // 输出数据 (output_height x output_width x output_channels)

    // 计算输出特征图大小
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // 对每个输出通道进行卷积
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                float sum = 0.0f;

                // 对每个输入通道
                for (int ic = 0; ic < input_channels; ic++) {
                    // 对卷积核的每个元素
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            // 计算输入数据中的对应位置
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            // 检查边界（padding区域填充0）
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = ih * input_width * input_channels + iw * input_channels + ic;
                                int kernel_idx = kh * kernel_width * input_channels * output_channels +
                                                 kw * input_channels * output_channels +
                                                 ic * output_channels + oc;
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }

                // 存储输出结果
                int output_idx = oh * output_width * output_channels + ow * output_channels + oc;
                output[output_idx] = sum;
            }
        }
    }
}

/**
 * 添加ReLU激活函数
 */
void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

/**
 * 添加偏置项
 */
void add_bias(float* data, int size, const float* bias, int channels) {
    for (int i = 0; i < size; i++) {
        int channel = i % channels;
        data[i] += bias[channel];
    }
}

/**
 * 打印张量数据（用于调试）
 */
void print_tensor(const float* data, int height, int width, int channels, const char* name) {
    printf("\n%s (Height=%d, Width=%d, Channels=%d):\n", name, height, width, channels);

    for (int c = 0; c < channels; c++) {
        printf("Channel %d:\n", c);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = h * width * channels + w * channels + c;
                printf("%8.3f ", data[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

/**
 * 测试卷积操作
 */
void test_convolution() {
    printf("=== 卷积操作测试 ===\n");

    // 定义输入数据 (4x4x1 - 单通道灰度图像)
    float input[4 * 4 * 1] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // 定义卷积核 (3x3x1x1 - 3x3卷积核，1输入通道，1输出通道)
    // 边缘检测卷积核（Sobel算子）
    float kernel[3 * 3 * 1 * 1] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    // 偏置项
    float bias[1] = {0};

    // 计算输出大小
    int input_height = 4, input_width = 4, input_channels = 1;
    int kernel_height = 3, kernel_width = 3, output_channels = 1;
    int stride = 1, padding = 0;
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // 分配输出内存
    float* output = (float*)malloc(output_height * output_width * output_channels * sizeof(float));

    // 执行卷积
    conv2d(input, input_height, input_width, input_channels,
           kernel, kernel_height, kernel_width, output_channels,
           stride, padding, output);

    // 添加偏置
    add_bias(output, output_height * output_width * output_channels, bias, output_channels);

    // 应用ReLU激活
    relu(output, output_height * output_width * output_channels);

    // 打印结果
    print_tensor(input, input_height, input_width, input_channels, "输入数据");
    print_tensor(kernel, kernel_height, kernel_width, input_channels, "卷积核");
    print_tensor(output, output_height, output_width, output_channels, "输出数据");

    // 清理内存
    free(output);
}

/**
 * 测试多通道卷积
 */
void test_multi_channel_convolution() {
    printf("\n=== 多通道卷积测试 ===\n");

    // 定义输入数据 (4x4x3 - RGB图像)
    float input[4 * 4 * 3] = {
        // R通道
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        // G通道
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        // B通道
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    // 定义卷积核 (3x3x3x2 - 3x3卷积核，3输入通道，2输出通道)
    float kernel[3 * 3 * 3 * 2] = {0};

    // 第一个输出通道的卷积核（边缘检测）
    float kernel1[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float kernel2[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float kernel3[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 第二个输出通道的卷积核（模糊）
    float kernel4[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};
    float kernel5[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};
    float kernel6[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};

    // 填充卷积核
    for (int i = 0; i < 9; i++) {
        kernel[i * 2 + 0] = kernel1[i];
        kernel[9 + i * 2 + 0] = kernel2[i];
        kernel[18 + i * 2 + 0] = kernel3[i];
        kernel[i * 2 + 1] = kernel4[i];
        kernel[9 + i * 2 + 1] = kernel5[i];
        kernel[18 + i * 2 + 1] = kernel6[i];
    }

    // 偏置项
    float bias[2] = {0, 0};

    // 参数
    int input_height = 4, input_width = 4, input_channels = 3;
    int kernel_height = 3, kernel_width = 3, output_channels = 2;
    int stride = 1, padding = 0;
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // 分配输出内存
    float* output = (float*)malloc(output_height * output_width * output_channels * sizeof(float));

    // 执行卷积
    conv2d(input, input_height, input_width, input_channels,
           kernel, kernel_height, kernel_width, output_channels,
           stride, padding, output);

    // 添加偏置和激活
    add_bias(output, output_height * output_width * output_channels, bias, output_channels);
    relu(output, output_height * output_width * output_channels);

    // 打印结果
    print_tensor(output, output_height, output_width, output_channels, "多通道输出");

    free(output);
}

int main() {
    test_convolution();
    test_multi_channel_convolution();

    printf("\n卷积操作实现完成！\n");
    return 0;
}
